import torch
from datasets.dataset import datamodule_gen, mil_collate_fn
from models.clam import CLAM_SB, CLAM_MB
from models.transmil import TransMIL
from models.mambamil import MambaMIL
from models.foundation_slide import FoundationSlideClassifier
from models.acmil import ACMILConfig, ACMILContinual
from models.wikg import WiKG
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
from utils import CustomEarlyStopping as EarlyStopping
from utils import CustomProgressBar
import yaml
import wandb
import os
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from torchmetrics import AUROC, Accuracy, MeanMetric
from lightning.fabric import Fabric, seed_everything, is_wrapped
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.loggers import CSVLogger
import copy
from collections import defaultdict
from dataclasses import replace
import random
import time
import math
import hashlib
import resource
from pathlib import Path
from sklearn.metrics import confusion_matrix
from experiment_logger import ExperimentLogger
from replay import build_replay_selector
import replay.mambamil_adapter  # register MambaMIL replay/region hooks
from replay.losses import replay_distillation_loss
from replay.stable_replay import calibrate_classification_head, stored_logit_loss
from replay.buffer_snapshot import save_buffer_snapshot
from replay.model_registry import get_training_loss
from replay.pseudobag_train import PseudoBagReplayEpoch
import replay.acmil_adapter  # noqa: F401 - registers ACMIL hooks
from replay.wikg_adapter import (
    continual_forward as wikg_continual_forward,
    supervised_loss as wikg_supervised_loss,
    validate_continual_config as validate_wikg_continual_config,
)
from replay.adapters import ReplayModelAdapter
from replay.global_budget_buffer import (
    AdmissionCandidate,
    GlobalPatchBudgetBuffer,
    load_patch_budget,
    ratio_patch_budget,
    deterministic_priority,
    candidate_replay_metadata,
)
from replay.adaptive_budget_buffer import AdaptiveGlobalPatchBuffer
from replay.unified_gradient_coverage import (
    ReplayOption,
    ReplaySlideCandidate,
    SelectedReplayOption,
    UnifiedGradientCoverageBuffer,
    deterministic_alternative_trajectory,
    model_state_fingerprint,
    replay_option_views,
)
from replay.region_supervision import RegionConfig, RegionSupervisor
from replay.conslide_buro import ConSlideBuRoBuffer
from replay.fixed_replay import FixedReplayEpoch, fixed_replay_target
from replay.full_wsi_buffer import FullWSIPatchBudgetBuffer


def _default_log_root():
    """Return the log root: $KEM_LOG_ROOT if set, otherwise ./logs."""
    override = os.environ.get('KEM_LOG_ROOT')
    if override:
        return str(Path(override).expanduser())
    return 'logs'


def _run_log_root(args):
    root = Path(args.log_root)
    group = str(getattr(args, 'log_group', '')).strip()
    if not group:
        return root
    group_path = Path(group)
    if group_path.is_absolute() or '..' in group_path.parts:
        raise ValueError('log_group must be a relative experiment name')
    return root / group_path / 'runs'


def _experiment_log_path(args, *parts):
    return _run_log_root(args) / args.exp_name / Path(*map(str, parts))


def compute_task_metrics(y_true, y_pred, y_prob, task_classes):
    """Compute task-level macro AUROC and macro F1 from global-class outputs."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    task_classes = np.asarray(task_classes, dtype=int)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=task_classes.tolist(),
        average='macro',
        zero_division=0,
    )

    class_aurocs = []
    for class_id in task_classes:
        binary_target = (y_true == class_id).astype(int)
        if np.unique(binary_target).size < 2:
            class_aurocs.append(np.nan)
        else:
            class_aurocs.append(roc_auc_score(binary_target, y_prob[:, class_id]))
    macro_auroc = np.nanmean(class_aurocs) if not np.all(np.isnan(class_aurocs)) else np.nan
    return float(macro_auroc), float(macro_f1)


def acmil_supervised_loss(output, label):
    return get_training_loss("acmil").compute(output, label)


def pseudobag_term_loss(args, model, features, labels, seen_classes, unseen_classes):
    """ER replay loss of one (pseudo-)bag; ``labels`` is [(label, weight)] (soft labels for mixup)."""
    if args.net == 'mambamil':
        out = model(features)
        out['logits'][:, unseen_classes] = -100
        return sum(w * F.cross_entropy(out['logits'], y) for y, w in labels)
    if args.net == 'acmil':
        out = model(features, seen_classes=seen_classes)
        return sum(w * acmil_supervised_loss(out, y)[0] for y, w in labels)
    if args.net in ['clam_sb', 'clam_mb']:
        dominant = max(labels, key=lambda item: item[1])[0]
        out = model(features, dominant, instance_eval=True, return_features=True, seen_classes=seen_classes)
        return sum(w * (0.7 * F.cross_entropy(out['logits'], y) + 0.3 * out['instance_loss']) for y, w in labels)
    raise NotImplementedError(f'pseudo-bag replay training is not defined for {args.net}')


def acmil_prev_distillation_loss(student, teacher, old_seen_classes):
    """Distill ACMIL bag, attention, and token outputs on old classes only."""
    old_classes = torch.as_tensor(
        old_seen_classes, device=student['logits'].device, dtype=torch.long
    )
    if old_classes.numel() == 0:
        raise ValueError('ACMIL prev distillation requires at least one old class')
    current = dict(student)
    previous = dict(teacher)
    for key in ('logits', 'token_logits'):
        current[key] = current[key].index_select(-1, old_classes)
        previous[key] = previous[key].index_select(-1, old_classes)
    return replay_distillation_loss('acmil', current, previous)


def normalize_acmil_heads(model):
    model.Slide_classifier.fc.weight.data = F.normalize(
        model.Slide_classifier.fc.weight.data
    )
    for head in model.classifier:
        head.fc.weight.data = F.normalize(head.fc.weight.data)


def evaluate_continual_loaders(model, loaders, args, fabric, seen_classes,
                                unseen_classes, current_task):
    """Evaluate all task loaders without using test results for model selection."""
    metrics = {}
    for idx, loader in enumerate(loaders):
        task_classes = np.asarray(np.unique(loader.dataset.targets), dtype=int)
        labels, predictions, probabilities = [], [], []
        for batch_idx, batch in enumerate(loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break
            batch = fabric.to_device(batch)
            with torch.no_grad():
                if args.net in ['clam_sb', 'clam_mb', 'acmil']:
                    output = model(batch['features'], seen_classes=seen_classes)
                elif args.net in ['transmil', 'mambamil', 'chief_cached']:
                    output = model(batch['features'])
                    output['logits'][:, unseen_classes] = -100
                elif args.net == 'wikg':
                    output = wikg_continual_forward(model, batch['features'], seen_classes)
                else:
                    raise NotImplementedError
            prediction = output['logits'].argmax(dim=-1)
            probability = F.softmax(output['logits'], dim=-1)
            labels.extend(batch['label'].detach().cpu().numpy().tolist())
            predictions.extend(prediction.detach().cpu().numpy().tolist())
            probabilities.append(probability.detach().cpu().numpy())
        accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels)))
        if idx > current_task:
            macro_auroc, macro_f1 = np.nan, 0.0
        else:
            macro_auroc, macro_f1 = compute_task_metrics(
                labels, predictions, np.concatenate(probabilities, axis=0), task_classes
            )
        metrics.update({
            f'{idx}_acc': accuracy,
            f'{idx}_auroc': macro_auroc,
            f'{idx}_f1': macro_f1,
        })
    return metrics


def _ratio_metadata(args):
    path = getattr(args, 'buffer_metadata_csv', None)
    if not path:
        path = os.path.join(args.data_root, 'tcga_morph_manifest.csv')
    frame = pd.read_csv(path)
    required = {'slide_id', 'n_patches'}
    if not required.issubset(frame.columns):
        raise ValueError(f'buffer metadata must contain {sorted(required)}: {path}')
    if frame.slide_id.duplicated().any():
        raise ValueError(f'duplicate slide_id in buffer metadata: {path}')
    return frame.set_index('slide_id')['n_patches'].astype(int).to_dict()


def _global_patch_mode(args):
    return args.buffer_mode in {'ratio_with_global_cap', 'adaptive_global_patch'}


def _full_wsi_mode(args):
    return getattr(args, 'buffer_mode', None) == 'strict_full_wsi'


def _conslide_buro_mode(args):
    return getattr(args, 'cl_method', None) == 'conslide_buro'


def _log_root(args):
    return str(_run_log_root(args))


def _experiment_path(args, *parts):
    return os.path.join(_log_root(args), str(args.exp_name), *map(str, parts))


def _effective_loader_updates(args, loader):
    count = len(loader)
    if int(args.n_batches) > 0:
        # The historical loop stops at ``batch_idx > n_batches``.
        count = min(count, int(args.n_batches) + 1)
    if count < 1:
        raise RuntimeError('training loader has no current-task updates')
    return count


def _assignment_weighted_fixed_replay(args):
    return bool(
        args.buffer_mode == 'adaptive_global_patch'
        and args.buffer_allocation_policy == 'unified_gradient_coverage'
        and args.unified_assignment_weighted_replay
    )


def _buffer_inventory(buffer):
    if buffer is None:
        return {'stored_patches': 0, 'stored_wsi': 0, 'stored_pseudobags': 0}
    if hasattr(buffer, 'records'):
        wsi = len({str(record.source_slide_id) for record in buffer.records})
        return {
            'stored_patches': int(buffer.n_patches_total),
            'stored_wsi': int(wsi),
            'stored_pseudobags': int(len(buffer.records)),
        }
    if hasattr(buffer, '_all_samples'):
        samples = list(buffer._all_samples())
    elif hasattr(buffer, 'buffers') and isinstance(buffer.buffers, dict):
        samples = [sample for values in buffer.buffers.values() for sample in values]
    else:
        samples = list(getattr(buffer, 'buffer', []))
    slide_ids = set()
    pseudobags = 0
    for sample in samples:
        slide_id = sample.get('slide_id', '')
        if isinstance(slide_id, (list, tuple)):
            slide_id = slide_id[0]
        slide_ids.add(str(slide_id))
        partitions = sample.get('pseudo_bag_ids')
        pseudobags += (
            int(torch.unique(torch.as_tensor(partitions)).numel())
            if partitions is not None else 1
        )
    return {
        'stored_patches': int(getattr(buffer, 'n_patches_total', 0)),
        'stored_wsi': len(slide_ids),
        'stored_pseudobags': int(pseudobags),
    }


def _validate_strict_budget(args, buffer, task):
    if not bool(getattr(args, 'strict_patch_budget', False)) or buffer is None:
        return
    actual = int(getattr(buffer, 'n_patches_total', 0))
    budget = int(args.buffer_patch_budget)
    if actual > budget:
        raise RuntimeError(
            f'strict replay budget exceeded at task {task}: {actual} > {budget}'
        )
    plan = getattr(buffer, 'last_plan', {}) or {}
    expanded = set((plan.get('ratio_gap_fill_by_slide') or {}).keys())
    retained = set(plan.get('retained_slide_ids') or [])
    if expanded & retained:
        raise RuntimeError(
            'strict protocol forbids expanding historical replay patches: '
            f'{sorted(expanded & retained)[:5]}'
        )


def _adaptive_levels(args, patch_count=None):
    raw = args.buffer_patch_levels
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split(',')
    levels = sorted({int(value) for value in values if str(value).strip()})
    if not levels or levels[0] < 1:
        raise ValueError('buffer_patch_levels must contain positive integers')
    if patch_count is None:
        return tuple(levels)
    return tuple(sorted({min(int(patch_count), value) for value in levels}))


def _buffer_attention(args, model, batch, seen_classes):
    with torch.no_grad():
        if args.net in ['clam_sb', 'clam_mb']:
            output = model(batch['features'], batch['label'], seen_classes=seen_classes)
            return output['A'].mean(dim=0, keepdim=True).reshape(1, -1)
        if args.net == 'acmil':
            output = model(batch['features'], seen_classes=seen_classes)
            return output['A'].reshape(1, -1)
        if args.net == 'mambamil':
            output = model(batch['features'])
            return output['A'].mean(dim=1).reshape(1, -1)
        if args.net == 'transmil':
            output = model(batch['features'], return_attn=True)
            epsilon = torch.finfo(output['attn1'].dtype).eps
            first = output['attn1'] / output['attn1'].max(dim=1, keepdim=True)[0].clamp_min(epsilon)
            second = output['attn2'] / output['attn2'].max(dim=1, keepdim=True)[0].clamp_min(epsilon)
            return (((second + 1) * (first + 1) / 4).mean(dim=0, keepdim=True).reshape(1, -1))
    raise ValueError(f'global patch buffer does not support model {args.net!r}')


def _buffer_coords(args, slide_id, features):
    config = RegionConfig(
        enabled=False, cache_dir=getattr(args, 'region_cache_dir', 'runs/kem_region_teacher_cache')
    )
    return RegionSupervisor(config, args).load_and_validate_coords(slide_id, features.detach().cpu())


def _new_adaptive_candidates(args, dataset, buffer, model, fabric, seen_classes,
                             task, replay_selector):
    started = time.perf_counter()
    if replay_selector.kem_v4 is None:
        raise ValueError('adaptive_global_patch requires distill_method=kem_v4')
    counts = _ratio_metadata(args)
    existing = {item.slide_id for item in buffer.existing_candidates()}
    candidates, seen = [], set()
    model.eval()
    for dataset_index, row in dataset.slide_data.reset_index(drop=True).iterrows():
        slide_id = str(row['slide_id'])
        if slide_id in seen or slide_id in existing:
            raise RuntimeError(f'duplicate or old slide in current adaptive pool: {slide_id}')
        seen.add(slide_id)
        sample = dataset[int(dataset_index)]
        batch = fabric.to_device(mil_collate_fn([sample]))
        original = int(batch['features'].shape[0])
        if slide_id not in counts or int(counts[slide_id]) != original:
            raise ValueError(f'canonical patch count mismatch for {slide_id}')
        attention = _buffer_attention(args, model, batch, seen_classes)
        coords = _buffer_coords(args, slide_id, batch['features'])
        levels = _adaptive_levels(args, original)
        analysis = replay_selector.kem_v4.analyze(
            model, batch['features'], batch['label'], seen_classes,
            max(levels), scores=attention, coords=coords,
        )
        levels, utilities = analysis.allocation_utilities(levels)
        if args.buffer_allocation_policy == 'uniform':
            maximum = max(levels)
            utilities = tuple(1.0 + math.sqrt(level / maximum) for level in levels)
        elif args.buffer_allocation_policy != 'adaptive':
            raise ValueError(
                f'unknown buffer_allocation_policy={args.buffer_allocation_policy!r}'
            )
        candidates.append(AdmissionCandidate(
            slide_id=slide_id, label=int(row['label']), original_patch_count=original,
            patch_budget=levels[0], source_task=int(task), dataset_index=int(dataset_index),
            allocation_levels=levels, allocation_utilities=utilities,
            bag_signature=tuple(analysis.bag_signature.tolist()),
        ))
        if (dataset_index + 1) % 25 == 0 or dataset_index + 1 == len(dataset.slide_data):
            elapsed = time.perf_counter() - started
            print(
                f'KEM v4 admission analysis: {dataset_index + 1}/'
                f'{len(dataset.slide_data)} WSI, {elapsed:.1f}s',
                flush=True,
            )

    # Reward WSI whose classifier-gradient direction is not already represented
    # by the same class.  This is inexpensive and keeps admission model-aware.
    if args.buffer_allocation_policy == 'uniform':
        elapsed = time.perf_counter() - started
        return candidates, {
            'kem_v4/admission_analysis_seconds': elapsed,
            'kem_v4/admission_analysis_seconds_per_wsi': elapsed / max(len(candidates), 1),
            'kem_v4/analyzed_wsi': float(len(candidates)),
        }
    adjusted = []
    for label in sorted({item.label for item in candidates}):
        group = [item for item in candidates if item.label == label]
        signatures = F.normalize(torch.tensor([item.bag_signature for item in group]), dim=1)
        centroid = F.normalize(signatures.mean(dim=0), dim=0)
        novelty = 1 - signatures @ centroid
        for item, value in zip(group, novelty.tolist()):
            adjusted.append(replace(
                item,
                allocation_utilities=tuple(float(score) + 0.25 * float(value)
                                           for score in item.allocation_utilities),
            ))
    elapsed = time.perf_counter() - started
    return adjusted, {
        'kem_v4/admission_analysis_seconds': elapsed,
        'kem_v4/admission_analysis_seconds_per_wsi': elapsed / max(len(adjusted), 1),
        'kem_v4/analyzed_wsi': float(len(adjusted)),
    }


def _unified_levels(args, patch_count):
    """Standard nested depths plus a derived exact-budget closure depth."""
    if getattr(args, 'unified_fixed_depth64', False):
        levels = {min(64, int(patch_count))}
        remainder = int(args.buffer_patch_budget) % 64
        if remainder and int(patch_count) >= 64 + remainder:
            levels.add(64 + remainder)
        return tuple(sorted(levels))
    levels = set(_adaptive_levels(args, patch_count))
    minimum = min(_adaptive_levels(args))
    remainder = int(args.buffer_patch_budget) % minimum
    if remainder and patch_count >= minimum + remainder:
        levels.add(minimum + remainder)
    return tuple(sorted(levels))


def _unified_candidate_rows(args, dataset, counts):
    rows = dataset.slide_data.reset_index(drop=True)
    if args.unified_candidate_max_slides > 0 and len(rows) > args.unified_candidate_max_slides:
        labels = sorted(rows.label.unique().tolist())
        base, remainder = divmod(int(args.unified_candidate_max_slides), len(labels))
        chosen_indices = []
        for offset, label in enumerate(labels):
            count = base + int(offset < remainder)
            group_indices = rows.index[rows.label == label].tolist()
            group_indices.sort(key=lambda index: (
                counts[str(rows.loc[index, 'slide_id'])],
                str(rows.loc[index, 'slide_id']),
            ))
            chosen_indices.extend(group_indices[:count])
        rows = rows.loc[sorted(chosen_indices)]
    return rows


def _new_unified_candidates(args, dataset, buffer, model, fabric, seen_classes,
                            task, replay_selector):
    """Build two deterministic nested option trajectories per current WSI."""
    started = time.perf_counter()
    if replay_selector.kem_v4 is None:
        raise ValueError('unified_gradient_coverage requires distill_method=kem_v4')
    counts = _ratio_metadata(args)
    existing = {item.slide_id for item in buffer.existing_slide_candidates()}
    candidate_rows = _unified_candidate_rows(args, dataset, counts)
    candidate_slide_ids = candidate_rows.slide_id.astype(str).tolist()

    def analyzed_patch_count(items):
        total = 0
        for item in items:
            maximum = max(int(option.patch_count) for option in item.options)
            total += min(
                int(item.original_patch_count),
                max(maximum, int(args.kem_v4_candidate_max)),
            )
        return float(total)
    cache_path = None
    if args.unified_candidate_cache and int(task) == 0:
        cache_fields = {
            'schema': 'kem-unified-options-v2', 'fold': int(buffer.fold),
            'task': int(task), 'seed': int(args.seed), 'net': args.net,
            'feature_extractor': args.feature_extractor, 'n_classes': int(args.n_classes),
            'projection_dim': int(args.unified_projection_dim),
            'signature_seed': int(args.unified_signature_seed),
            'patch_levels': str(args.buffer_patch_levels),
            'patch_budget': int(args.buffer_patch_budget),
            'region_candidates': bool(args.unified_region_stratified_candidates),
            'region_weight': float(args.kem_v4_region_weight),
            'influence_weight': float(args.kem_v4_influence_weight),
            'leverage_weight': float(args.kem_v4_leverage_weight),
            'diversity_weight': float(args.kem_v4_diversity_weight),
            'attention_weight': float(args.kem_v4_attention_weight),
            'candidate_max': int(args.kem_v4_candidate_max),
            'candidate_pool': str(args.kem_v4_candidate_pool),
            'selection_rule': str(args.kem_v4_selection_rule),
            'trajectory_mode': str(args.unified_trajectory_mode),
            'signature_mode': str(args.unified_frozen_signature_mode),
            'fixed_depth64': bool(args.unified_fixed_depth64),
            'class_conditioned': bool(args.unified_class_conditioned),
            'assignment_view': str(args.unified_assignment_view),
            'assignment_weighted': bool(args.unified_assignment_weighted_replay),
            'candidate_slide_ids': candidate_slide_ids,
            'model_state': model_state_fingerprint(model),
        }
        digest = hashlib.sha256(json.dumps(
            cache_fields, sort_keys=True
        ).encode('utf-8')).hexdigest()[:20]
        cache_path = os.path.join(
            args.unified_candidate_cache_dir, f'fold{buffer.fold}_task0_{digest}.pt'
        )
        if os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location='cpu', weights_only=False)
            if payload.get('cache_fields') != cache_fields:
                raise RuntimeError(f'unified candidate cache fingerprint mismatch: {cache_path}')
            candidates = payload['candidates']
            if [item.slide_id for item in candidates] != candidate_slide_ids:
                raise RuntimeError(f'unified candidate cache dataset mismatch: {cache_path}')
            elapsed = time.perf_counter() - started
            return candidates, {
                'unified/option_analysis_seconds': elapsed,
                'unified/option_analysis_seconds_per_wsi': elapsed / max(len(candidates), 1),
                'unified/analyzed_wsi': float(len(candidates)),
                'unified/analyzed_patch_count': analyzed_patch_count(candidates),
                'unified/options': float(sum(len(item.options) for item in candidates)),
                'unified/candidate_cache_hit': 1.0,
            }
    candidates, seen = [], set()
    model.eval()
    for progress, (dataset_index, row) in enumerate(candidate_rows.iterrows(), start=1):
        slide_id = str(row['slide_id'])
        if slide_id in seen or slide_id in existing:
            raise RuntimeError(f'duplicate or old slide in current unified pool: {slide_id}')
        seen.add(slide_id)
        sample = dataset[int(dataset_index)]
        batch = fabric.to_device(mil_collate_fn([sample]))
        original = int(batch['features'].shape[0])
        if slide_id not in counts or int(counts[slide_id]) != original:
            raise ValueError(f'canonical patch count mismatch for {slide_id}')
        attention = _buffer_attention(args, model, batch, seen_classes)
        coords = (
            _buffer_coords(args, slide_id, batch['features'])
            if args.unified_region_stratified_candidates else None
        )
        levels = _unified_levels(args, original)
        maximum = max(levels)
        analysis = replay_selector.kem_v4.analyze(
            model, batch['features'], batch['label'], seen_classes,
            maximum, scores=attention, coords=coords,
            deterministic_key=f'{buffer.fold}|{slide_id}',
        )
        primary = analysis.ranking[:maximum].cpu().long()
        alternative = None
        if args.unified_trajectory_mode in {'b', 'ab'}:
            alternative = deterministic_alternative_trajectory(
                batch['features'], primary, attention, maximum,
                args.seed, slide_id,
            )
        signature_seed = int(args.unified_signature_seed)
        reference_views = replay_option_views(
            model, args.net, batch['features'], batch['label'], seen_classes,
            args.n_classes, args.unified_projection_dim, signature_seed,
            args.unified_frozen_signature_mode,
        )
        _, scalar_utilities = analysis.allocation_utilities(levels)
        scalar_by_depth = dict(zip(levels, scalar_utilities))
        options = []
        trajectories = []
        if args.unified_trajectory_mode in {'a', 'ab'}:
            trajectories.append(('a', primary))
        if args.unified_trajectory_mode in {'b', 'ab'}:
            trajectories.append(('b', alternative))
        for trajectory_name, trajectory in trajectories:
            for depth in levels:
                indices = trajectory[:depth]
                views = replay_option_views(
                    model, args.net,
                    batch['features'].index_select(0, indices.to(batch['features'].device)),
                    batch['label'], seen_classes, args.n_classes,
                    args.unified_projection_dim, signature_seed,
                    args.unified_frozen_signature_mode,
                )
                views['v4_scalar'] = torch.tensor([scalar_by_depth[depth]])
                if args.unified_fixed_depth64 and depth != min(64, original):
                    views['exact_closure_only'] = torch.ones(1)
                options.append(ReplayOption(
                    slide_id=slide_id, label=int(row['label']), source_task=int(task),
                    trajectory=trajectory_name, patch_indices=indices,
                    signature=views['gradient'], dataset_index=int(dataset_index),
                    is_existing=False,
                    views=views,
                ))
        candidates.append(ReplaySlideCandidate(
            slide_id=slide_id, label=int(row['label']), source_task=int(task),
            original_patch_count=original, reference_signature=reference_views['gradient'],
            options=tuple(options), dataset_index=int(dataset_index), is_existing=False,
            reference_views=reference_views,
        ))
        if progress % 25 == 0 or progress == len(candidate_rows):
            print(
                f'Unified option analysis: {progress}/{len(candidate_rows)} WSI, '
                f'{time.perf_counter() - started:.1f}s', flush=True,
            )
    elapsed = time.perf_counter() - started
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        temporary = f'{cache_path}.{os.getpid()}.tmp'
        torch.save({'cache_fields': cache_fields, 'candidates': candidates}, temporary)
        os.replace(temporary, cache_path)
    return candidates, {
        'unified/option_analysis_seconds': elapsed,
        'unified/option_analysis_seconds_per_wsi': elapsed / max(len(candidates), 1),
        'unified/analyzed_wsi': float(len(candidates)),
        'unified/analyzed_patch_count': analyzed_patch_count(candidates),
        'unified/options': float(sum(len(item.options) for item in candidates)),
        'unified/candidate_cache_hit': 0.0,
    }


def _new_ratio_candidates(args, dataset, buffer, fold, task):
    counts = _ratio_metadata(args)
    existing = {item.slide_id for item in buffer.existing_candidates()}
    candidates, seen = [], set()
    for dataset_index, row in dataset.slide_data.reset_index(drop=True).iterrows():
        slide_id = str(row['slide_id'])
        if slide_id in seen:
            raise ValueError(f'duplicate slide in task dataset: {slide_id}')
        seen.add(slide_id)
        if slide_id in existing:
            raise RuntimeError(f'strict CL violation: current dataset contains old buffered slide {slide_id}')
        if slide_id not in counts:
            raise KeyError(f'missing original patch count for {slide_id}')
        original = int(counts[slide_id])
        candidates.append(AdmissionCandidate(
            slide_id=slide_id,
            label=int(row['label']),
            original_patch_count=original,
            patch_budget=ratio_patch_budget(original, args.buffer_slide_ratio, args.buffer_ratio_floor),
            source_task=int(task),
            dataset_index=int(dataset_index),
            is_existing=False,
        ))
    return candidates


def _select_buffer_sample(args, sample, patch_budget, replay_selector, model, fabric,
                          seen_classes, task, observability, candidate=None):
    batch = mil_collate_fn([sample])
    original_patch_count = int(batch['features'].size(0))
    if original_patch_count != int(sample.get('n_patch', original_patch_count)):
        raise ValueError('dataset patch count changed while constructing ratio buffer')
    if patch_budget > original_patch_count:
        raise ValueError('per-WSI patch budget exceeds original bag size')
    batch = fabric.to_device(batch)
    patch_selector = (args.buffer_patch_selector
                      if args.buffer_mode == 'adaptive_global_patch'
                      else args.distill_method)
    attn = None if patch_selector == 'random' else _buffer_attention(
        args, model, batch, seen_classes
    )
    coords = (_buffer_coords(args, str(sample['slide_id']), batch['features'])
              if patch_selector in {'kem_v4', 'spatial'} else None)
    if isinstance(candidate, SelectedReplayOption):
        selection_started = time.perf_counter()
        indices = candidate.selected_indices
        if indices.numel() != int(patch_budget):
            raise AssertionError('cached unified indices do not match the planned depth')
        selection = type('Selection', (), {
            'indices': indices,
            'partition_ids': None,
            'diagnostics': {
                'unified/cached_materialization': 1.0,
                'unified/assignment_count': float(candidate.assignment_count),
                'unified/assignment_mass': float(candidate.assignment_mass),
                'unified/marginal_coverage': float(candidate.marginal_coverage),
                'selector/selection_time_seconds': time.perf_counter() - selection_started,
            },
        })()
    elif patch_selector == 'random':
        selection_started = time.perf_counter()
        priority = deterministic_priority(args.seed, 0, task, str(sample['slide_id']))
        generator = torch.Generator(device=batch['features'].device).manual_seed(
            args.seed + priority % (2**31)
        )
        indices = torch.randperm(
            original_patch_count, generator=generator,
            device=batch['features'].device,
        )[:int(patch_budget)].cpu()
        selection = type('Selection', (), {
            'indices': indices,
            'partition_ids': None,
            'diagnostics': {
                'buffer/random_patch_selector': 1.0,
                'selector/selection_time_seconds': time.perf_counter() - selection_started,
            },
        })()
    else:
        selection = replay_selector.select(
            features=batch['features'], budget=int(patch_budget), scores=attn,
            model=model, label=batch['label'], seen_classes=seen_classes, coords=coords,
            slide_id=str(sample['slide_id']), task=task,
        )
    if int(selection.indices.numel()) != int(patch_budget):
        raise AssertionError('selector did not honor the planned whole-WSI patch budget')
    batch['features'] = batch['features'].index_select(0, selection.indices.to(batch['features'].device))
    batch['selected_indices'] = selection.indices.detach().clone()
    selected_unified_views = None
    if (isinstance(candidate, SelectedReplayOption)
            and 'budget_closure_approximation' in candidate.option.views):
        exact_views = replay_option_views(
            model, args.net, batch['features'], batch['label'], seen_classes,
            args.n_classes, args.unified_projection_dim, args.unified_signature_seed,
            args.unified_frozen_signature_mode,
        )
        # Recompute every model-derived view at the exact closure depth while
        # retaining allocator-only metadata (notably the v4 scalar utility).
        # The latter has no model-forward equivalent and remains explicitly
        # marked as a closure approximation in the stored option.
        selected_unified_views = {
            **candidate.option.views,
            **exact_views,
        }
    if getattr(selection, 'partition_ids', None) is not None:
        partition_ids = selection.partition_ids.detach().cpu().long().reshape(-1)
        if partition_ids.numel() != selection.indices.numel():
            raise AssertionError('pseudo-bag partition IDs must align with selected indices')
        batch['pseudo_bag_ids'] = partition_ids
    batch['original_patch_count'] = torch.tensor(original_patch_count, device=batch['features'].device)
    batch['selection_method'] = patch_selector
    batch['source_task'] = torch.tensor(task, device=batch['features'].device)
    batch['ratio_requested'] = torch.tensor(args.buffer_slide_ratio, device=batch['features'].device)
    batch['ratio_actual'] = torch.tensor(patch_budget / original_patch_count, device=batch['features'].device)
    batch['ratio_floor_hit'] = torch.tensor(
        int(original_patch_count >= args.buffer_ratio_floor and patch_budget == args.buffer_ratio_floor),
        device=batch['features'].device,
    )
    batch['ratio_small_bag'] = torch.tensor(
        int(original_patch_count < args.buffer_ratio_floor), device=batch['features'].device,
    )
    if isinstance(candidate, SelectedReplayOption):
        selected_views = selected_unified_views or candidate.option.views
        option_signatures = {
            str(option.patch_count): option.signature.to(batch['features'].device)
            for option in candidate.prefix_options
        }
        option_views = {
            str(option.patch_count): {
                key: value.to(batch['features'].device)
                for key, value in option.views.items()
            } for option in candidate.prefix_options
        }
        if selected_unified_views is not None:
            depth_key = str(candidate.patch_budget)
            option_signatures[depth_key] = selected_unified_views['gradient'].to(
                batch['features'].device
            )
            option_views[depth_key] = {
                key: value.to(batch['features'].device)
                for key, value in selected_unified_views.items()
            }
        batch.update({
            'reference_gradient_signature': candidate.reference_signature.to(
                batch['features'].device
            ),
            'unified_reference_views': {
                key: value.to(batch['features'].device)
                for key, value in candidate.reference_views.items()
            },
            'unified_selected_signature': selected_views['gradient'].to(
                batch['features'].device
            ),
            'unified_trajectory': candidate.option.trajectory,
            'unified_selected_views': {
                key: value.to(batch['features'].device)
                for key, value in selected_views.items()
            },
            'unified_option_signatures': option_signatures,
            'unified_option_indices': {
                str(option.patch_count): option.patch_indices.to(batch['features'].device)
                for option in candidate.prefix_options
            },
            'unified_option_views': option_views,
            'unified_assignment_count': torch.tensor(
                candidate.assignment_count, device=batch['features'].device,
            ),
            'unified_assignment_mass': torch.tensor(
                candidate.assignment_mass, device=batch['features'].device,
            ),
            'allocation_utility': torch.tensor(
                candidate.marginal_coverage, device=batch['features'].device,
            ),
            'bag_signature': candidate.reference_signature.to(batch['features'].device),
        })
    else:
        batch.update(candidate_replay_metadata(candidate, batch['features'].device))
    if patch_selector == 'kem_v4' and args.buffer_region_lambda > 0:
        selected_coords = coords.index_select(0, selection.indices.cpu()) if coords is not None else None
        targets = replay_selector.kem_v4.build_region_targets(
            model, batch['features'], seen_classes, selected_coords,
        )
        batch.update({name: value.to(batch['features'].device)
                      for name, value in targets.items()})
    if selection.diagnostics:
        observability.log_scalars(selection.diagnostics)
        batch['selection_diagnostics'] = dict(selection.diagnostics)
    if args.cl_method == 'derpp' or args.buffer_mode == 'adaptive_global_patch':
        with torch.no_grad():
            output = ReplayModelAdapter(args.net).forward(
                model, batch['features'], seen_classes
            )
        batch['logits'] = output.logits.detach().clone()
        batch['stored_embedding'] = output.features.detach().clone()
        batch['stored_classes'] = torch.as_tensor(
            seen_classes, device=output.logits.device, dtype=torch.long
        )
    return batch

# use pure pytorch instead of pytorch-lightning
class SimpleBuffer():
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffer = []  # List of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples = 0
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for i in range(len(self.buffer)):
                self.buffer[i] = move_data_to_device(self.buffer[i], device)
        return self

    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        # In this simple version, we do not adjust the buffer size
        pass

    def add_judge(self, sample_label):
        """Determine whether to add a new sample to the buffer using reservoir sampling."""
        n_seen = self.n_seen_samples
        self.n_seen_samples += 1
        if len(self.buffer) < self.buffer_size:
            # Buffer not full yet
            return len(self.buffer)
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size:
                return rand_idx
            else:
                return -1  # Do not add the sample

    def add(self, sample, idx):
        """Add a sample to the buffer at the given index."""
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffer):
            # Adding a new sample to the buffer
            self.buffer.append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            # Replacing an existing sample in the buffer
            self.n_patches_total = self.n_patches_total - self.buffer[idx]['features'].size(0) + sample['features'].size(0)
            old_label = self.buffer[idx]['label'].item()
            self.labels[old_label] -= 1
            self.buffer[idx] = sample
            self.labels[label] = self.labels.get(label, 0) + 1

    def start_epoch(self):
        """Reset indices for a new epoch."""
        self.epoch_indices = np.random.permutation(len(self.buffer))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get the next batch of samples for the current epoch."""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.buffer) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.buffer[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch."""
        return self.samples_output_in_epoch

    def __len__(self):
        return len(self.buffer)
    

class ClsssIncrementalBuffer():
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffers = {}  # Dict mapping class labels to lists of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples_per_class = defaultdict(int)
        self.num_classes = 0
        self.buffer_size_per_class = None
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for l in self.buffers:
                for i in range(len(self.buffers[l])):
                    self.buffers[l][i] = move_data_to_device(self.buffers[l][i], device)
        return self

    # Given the number of new classes will be adding, adjust the buffer size per class
    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        self.buffer_size_per_class = self.buffer_size // (self.num_classes + n_new_classes)
        for l in self.buffers:
            if len(self.buffers[l]) > self.buffer_size_per_class:
                self.buffers[l] = random.sample(self.buffers[l], self.buffer_size_per_class)
                self.labels[l] = self.buffer_size_per_class
        # calculate the number of patches in the buffer
        self.n_patches_total = sum([sample['features'].size(0) for samples in self.buffers.values() for sample in samples])

    # Judge whether to add the new sample to the buffer by reservoir sampling
    # If add, return the index
    def add_judge(self, sample_label):
        if sample_label not in self.buffers:
            # New class encountered
            self.buffers[sample_label] = []
            self.num_classes += 1

        self.n_seen_samples_per_class[sample_label] += 1
        n_seen = self.n_seen_samples_per_class[sample_label]
        if len(self.buffers[sample_label]) < self.buffer_size_per_class:
            return len(self.buffers[sample_label])
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size_per_class:
                return rand_idx
            else:
                return -1
            
    def add(self, sample, idx):
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffers[label]):
            self.buffers[label].append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            self.n_patches_total = self.n_patches_total - self.buffers[label][idx]['features'].size(0) + sample['features'].size(0)
            self.buffers[label][idx] = sample
            # Labels count remains the same

    def start_epoch(self):
        """Reset indices for a new epoch"""
        # Flatten buffers into a single list
        self.all_samples = []
        for samples in self.buffers.values():
            self.all_samples.extend(samples)
        self.epoch_indices = np.random.permutation(len(self.all_samples))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get next batch of samples for the current epoch"""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.all_samples) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.all_samples[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch"""
        return self.samples_output_in_epoch

    def __len__(self):
        return sum(len(samples) for samples in self.buffers.values())

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

    
def add_argument(parser, name, value):
    """Helper function to add an argument to the parser if it doesn't already exist."""
    if not any(arg.dest == name for arg in parser._actions):
        if isinstance(value, bool):
            parser.add_argument(f'--{name}', action='store_false' if value else 'store_true', default=value)
        else:
            parser.add_argument(f'--{name}', type=type(value), default=value)

def load_config_from_yaml(file_path):
    """Loads configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file) or {}

def init_args():
    parser = argparse.ArgumentParser(description='MIL-CL')
    parser.add_argument('--preset', type=str, default=None, help='preset config')

    args, remaining_argv = parser.parse_known_args()

    # Load configuration from YAML if preset is provided
    config = load_config_from_yaml(args.preset) if args.preset else {}

    logging_defaults = {
        'log_root': _default_log_root(),
        'log_group': os.environ.get('KEM_LOG_GROUP', ''),
        'tensorboard': True,
        'wandb_mode': 'offline',
        'log_every_n_steps': 1,
        'log_histograms': True,
        'histogram_every_n_epochs': 1,
        'log_cl_matrices': True,
        'log_confusion_matrices': True,
        'project': 'KEM',
        'replay_schedule': 'natural',
        'replay_updates_per_epoch': 78,
        'fixed_epoch_training': False,
        'strict_patch_budget': False,
        'bilevel_solver': 'unrolled',
        'bilevel_proxy_scope': 'aggregator_head',
        'bilevel_prefilter': 'maxrand',
        'bilevel_coarse_ratio': 2.0,
        'bilevel_candidate_max': 2048,
        'bilevel_num_groups': 256,
        'bilevel_cluster_dim': 64,
        'bilevel_cluster_steps': 5,
        'bilevel_outer_steps': 3,
        'bilevel_inner_steps': 1,
        'bilevel_proxy_lr': 1e-3,
        'bilevel_weight_lr': 0.1,
        'bilevel_distill_lambda': 0.1,
        'bilevel_feature_lambda': 0.1,
        'bilevel_temperature': 0.1,
        'bilevel_neumann_steps': 3,
        'bilevel_damping': 0.1,
        'bilevel_nystrom_rank': 5,
        'bilevel_bome_eta': 0.5,
        'kem_variant': 'anchored',
        'kem_anchor_ratio': 0.5,
        'kem_anchor_method': 'maxminrand',
        'kem_candidate_method': 'maxminrand',
        'kem_candidate_ratio': 4.0,
        'kem_backend': 'auto',
        'kem_outer_steps': 15,
        'kem_weight_lr': 0.03,
        'kem_temperature_start': 0.1,
        'kem_temperature_end': 0.01,
        'kem_gradient_lambda': 1.0,
        'kem_diversity_lambda': 0.05,
        'stored_logits_lambda': 1.0,
        'head_calibration': False,
        'head_calibration_epochs': 3,
        'head_calibration_lr': 0.0001,
        'head_calibration_weight_norm': True,
        'save_buffer_snapshots': False,
        'buffer_snapshot_dir': 'buffer_snapshots',
        'buffer_snapshot_fp16': False,
        'buffer_snapshot_full': False,
        'region_supervision': False,
        'region_partition': 'spatial_balanced_100',
        'region_target_patches': 100,
        'region_fixed_count': 10,
        'region_teacher_temperature': 2.0,
        'region_confidence_power': 2.0,
        'region_loss_lambda': 0.5,
        'region_min_size': 16,
        'region_cache_dir': 'runs/kem_region_teacher_cache',
        'acmil_n_token': 5,
        'acmil_attention_dim': 128,
        'acmil_n_masked_patch': 10,
        'acmil_mask_drop': 0.6,
        'acmil_droprate': 0.0,
        'buffer_mode': 'legacy',
        'buffer_slide_ratio': 0.05,
        'buffer_ratio_floor': 64,
        'buffer_budget_file': 'configs/tcga_morph_3t6c_5pct_patch_budgets.json',
        'buffer_budget_tolerance': 0.015,
        'buffer_metadata_csv': '',
        'buffer_capacity_policy': 'error',
        'buffer_patch_budget': 30000,
        'buffer_patch_levels': '64,128,256,512',
        'buffer_allocation_policy': 'adaptive',
        'buffer_patch_selector': 'kem_v4',
        'buffer_embedding_lambda': 0.05,
        'buffer_region_lambda': 0.1,
        'kem_v4_candidate_max': 2048,
        'kem_v4_projection_dim': 64,
        'kem_v4_max_regions': 32,
        'kem_v4_min_region_size': 16,
        'kem_v4_patch_dropout': 0.2,
        'kem_v4_coverage_ratio': 0.7,
        'kem_v4_influence_weight': 0.3,
        'kem_v4_leverage_weight': 0.25,
        'kem_v4_diversity_weight': 0.35,
        'kem_v4_region_weight': 0.2,
        'kem_v4_attention_weight': 0.1,
        'kem_v4_candidate_pool': 'maxminrand',
        'kem_v4_selection_rule': 'weighted_sum',
        'unified_projection_dim': 64,
        'unified_signature_seed': 1701,
        'unified_objective': 'class_gradient_coverage',
        'unified_worst_class_lexicographic': False,
        'unified_region_stratified_candidates': True,
        'unified_assignment_weighted_replay': False,
        'unified_trajectory_mode': 'ab',
        'unified_assignment_view': 'auto',
        'unified_class_conditioned': True,
        'unified_frozen_signature_mode': 'mean_std',
        'unified_fixed_depth64': False,
        'unified_candidate_cache': True,
        'unified_candidate_cache_dir': 'runs/kem_unified_candidate_cache',
        'unified_candidate_max_slides': -1,
        'mmil_num_subbags': 4,
        'mmil_group_max_size': 4300,
        'mmil_kmeans_steps': 8,
        'mmil_cluster_dim': 64,
        'prodiv_num_strata': 4,
        'psemix_num_phenotypes': 8,
        'psemix_refine_steps': 8,
        'pmil_m_max': 10,
        'pmil_search_rate': 10,
        'pmil_subset_count': 3,
        'pmil_timeout_seconds': 60.0,
        'replay_pseudobag_train': 'none',
        'pbtrain_min_bag': 2,
        'pbtrain_pmil_m_max': 10,
        'pbtrain_pmil_step': 2,
        'pbtrain_pmil_progressive': True,
        'pbtrain_prodiv_groups': 6,
        'pbtrain_psemix_n': 10,
        'pbtrain_psemix_prob': 0.4,
        'pbtrain_psemix_alpha': 1.0,
        'pbtrain_mmil_full_weight': 0.5,
        'conslide_region_size': 64,
        'conslide_store_ratio': 0.2,
        'conslide_replay_regions': 8,
        'conslide_region_partition': 'spatial',
        'conslide_buffer_max_slides': -1,
    }
    for key, value in logging_defaults.items():
        config.setdefault(key, value)

    # Update parser with options from YAML configuration
    for k, v in config.items():
        add_argument(parser, k, v)

    args = parser.parse_args()

    # update args
    if args.debug == 'False':
        args.debug = False
    elif args.debug == 'True':
        args.debug = 'lite'
    if args.debug == 'lite':
        args.n_batches = 10
        args.epochs = 2
        args.n_folds = 1
        args.wandb_mode = 'disabled'

        if hasattr(args, 'n_tasks'):
            args.n_tasks = 2
    elif args.debug == 'full':
        args.n_batches = 20
        args.epochs = 3
        args.wandb_mode = 'disabled'


    # ExperimentLogger initializes W&B lazily.  It never calls login or sync.
    args.log_root = str(Path(args.log_root).expanduser().resolve())
    os.environ['WANDB_MODE'] = args.wandb_mode

    if args.folds_start == -1:
        args.folds_start = 0
    if args.folds_end == -1:
        if hasattr(args, 'n_folds'):
            args.folds_end = args.n_folds
        else:
            args.folds_end = 1

    print(args)
    return args

def load_model(args):
    # if args.net == 'abmil':
        # model = Attn_Net(args)
    # elif args.net == 'abmil_gated':
        # model = Attn_Net_Gated(args)
    # elif args.net == 'dtfd':
        # model = DTFD_CL(args)
    if args.net == 'clam_sb':
        model = CLAM_SB(D_feat=args.D_feat,
                        L=args.L,
                        K=args.K,
                        dropout=args.dropout,
                        gate=args.gate,
                        k_sample=args.k_sample,
                        n_classes=args.n_classes,
                        instance_loss_name=args.instance_loss_name,
                        subtyping=args.subtyping,)
    elif args.net == 'clam_mb':
        model = CLAM_MB(D_feat=args.D_feat,
                        L=args.L,
                        K=args.K,
                        dropout=args.dropout,
                        gate=args.gate,
                        k_sample=args.k_sample,
                        n_classes=args.n_classes,
                        instance_loss_name=args.instance_loss_name,
                        subtyping=args.subtyping)
    elif args.net == 'acmil':
        config = ACMILConfig(
            n_class=args.n_classes,
            n_token=args.acmil_n_token,
            D_feat=args.D_feat,
            D_inner=args.L,
        )
        model = ACMILContinual(
            config,
            attention_dim=args.acmil_attention_dim,
            droprate=args.acmil_droprate,
            n_token=args.acmil_n_token,
            n_masked_patch=args.acmil_n_masked_patch,
            mask_drop=args.acmil_mask_drop,
        )
    elif args.net == 'transmil':
        model = TransMIL(D_feat=args.D_feat,
                         D_inner=args.L,
                         n_classes=args.n_classes,)
    elif args.net == 'mambamil':
        model = MambaMIL(
            in_dim=args.D_feat,
            n_classes=args.n_classes,
            dropout=args.mamba_dropout,
            act='gelu',
            layer=args.mamba_layers,
            rate=args.mamba_rate,
            mamba_type='SRMamba',
        )
    elif args.net == 'wikg':
        model = WiKG(
            dim_in=args.D_feat,
            dim_hidden=getattr(args, 'wikg_dim_hidden', 512),
            topk=getattr(args, 'wikg_topk', 6),
            n_classes=args.n_classes,
            agg_type=getattr(args, 'wikg_agg_type', 'bi-interaction'),
            dropout=getattr(args, 'wikg_dropout', 0.3),
            pool=getattr(args, 'wikg_pool', 'mean'),
        )
    elif args.net == 'chief_cached':
        model = FoundationSlideClassifier(
            n_classes=args.n_classes,
            input_dim=args.D_feat,
            dropout=getattr(args, 'foundation_dropout', 0.0),
        )
    else:
        raise NotImplementedError

    return model

def kd_loss_fn(train_logits, prev_logits, ta = 2, softmax = True):
    """
    KD_LOSS: Compute distillation loss between output of the current model and the output of the previous (saved) model.
        + Inputs:
            - train_logits: Logits of model in training phase only for active units
            - prev_logits: Logits of model previous experience only for active units
        + Outputs:
            - dist_loss: Knowledge distrillation loss
    """
    assert prev_logits.size() == train_logits.size()
    if softmax:
        q = torch.softmax(prev_logits / ta, dim = -1)
        log_p = torch.log_softmax(train_logits / ta, dim = -1)
    else:
        q = prev_logits
        log_p = torch.log(train_logits)
    dist_loss = torch.nn.functional.kl_div(log_p, q, reduction = "batchmean")
    return dist_loss



def one_fold(args, fold=0):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; CPU fallback is disabled')
    if args.replay_schedule not in {'natural', 'fixed'}:
        raise ValueError('replay_schedule must be natural or fixed')
    if int(args.replay_updates_per_epoch) < 0:
        raise ValueError('replay_updates_per_epoch must be non-negative')
    if args.replay_schedule == 'fixed' and not args.fixed_epoch_training:
        raise ValueError('fixed replay requires --fixed_epoch_training for protocol parity')
    if bool(args.strict_patch_budget) and int(args.buffer_patch_budget) < 1:
        raise ValueError('strict_patch_budget requires a positive buffer_patch_budget')
    if args.buffer_allocation_policy == 'unified_gradient_coverage':
        if args.buffer_mode != 'adaptive_global_patch':
            raise ValueError('unified_gradient_coverage requires buffer_mode=adaptive_global_patch')
        if args.distill_method != 'kem_v4' or args.buffer_patch_selector != 'kem_v4':
            raise ValueError('unified_gradient_coverage reuses the deterministic KEM v4 candidate generator')
        unified_objectives = {
            'random_options', 'prediction_fidelity', 'embedding_fidelity',
            'frozen_feature_coverage', 'class_mean_gradient_matching',
            'class_gradient_coverage', 'fisher_logdet', 'decision_boundary',
            'v4_scalar',
            'frozen_gradient_geomean_coverage',
        }
        if args.unified_objective not in unified_objectives:
            raise ValueError(f'unknown unified_objective={args.unified_objective!r}')
        if args.unified_trajectory_mode not in {'a', 'b', 'ab'}:
            raise ValueError('unified_trajectory_mode must be one of: a, b, ab')
        if (args.unified_worst_class_lexicographic
                and args.unified_objective != 'class_gradient_coverage'):
            raise ValueError('worst-class lexicographic is only defined for class_gradient_coverage')
    if _conslide_buro_mode(args):
        if args.net != 'clam_sb':
            raise ValueError(
                'ConSlide-BuRo is a CLAM-SB replay adaptation. Full ConSlide '
                'requires the HIT backbone and paired patch/region features; '
                f'net={args.net!r} is unsupported.'
            )
        if args.buffer_mode != 'conslide_buro':
            raise ValueError('cl_method=conslide_buro requires buffer_mode=conslide_buro')
        if args.cl_on_current:
            raise ValueError('ConSlide-BuRo uses replay loss, not current-bag distillation')
        if args.conslide_region_partition != 'spatial':
            raise ValueError('formal ConSlide-BuRo requires spatial region partitioning')
    if args.net == 'acmil' and getattr(args, 'cl_method', 'ER') not in {
        'ER', 'derpp', 'LwF', 'prev', 'none'
    }:
        raise ValueError('ACMIL continual training currently supports ER, derpp, LwF, prev, and none')
    if args.net == 'wikg':
        validate_wikg_continual_config(args)
    if args.net == 'chief_cached' and getattr(args, 'cl_method', 'none') in ['MICIL', 'prev']:
        raise ValueError(
            'chief_cached has no patch tokens/attention; MICIL and prev are unsupported. '
            'Use none, LwF, ER, derpp, or joint.'
        )
    fabric = Fabric(devices=1, accelerator="cuda")
    seed_everything(args.seed)
    observability = ExperimentLogger(args, fold, root=_run_log_root(args))
    replay_selector = build_replay_selector(args.distill_method, args.net, args)

    model = load_model(args)
    model = fabric.to_device(model)

    # Prepare test dataloaders from all tasks
    evaluation_modules = [datamodule_gen(args, fold=fold, task=task) for task in range(args.n_tasks)]
    test_dataloaders = [module['test_loader'] for module in evaluation_modules]
    validation_dataloaders = [module['val_loader'] for module in evaluation_modules]
    # test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    # memory pool setup
    buffer = None
    if hasattr(args, 'cl_method') and args.cl_method != 'joint':
        if hasattr(args, 'buffer_size') and args.buffer_size > 0:
            if _conslide_buro_mode(args):
                buffer = ConSlideBuRoBuffer(
                    patch_budget=args.buffer_patch_budget,
                    seed=args.seed,
                    fold=fold,
                    region_size=args.conslide_region_size,
                    store_ratio=args.conslide_store_ratio,
                    replay_regions=args.conslide_replay_regions,
                    partition=args.conslide_region_partition,
                )
            elif _full_wsi_mode(args):
                buffer = FullWSIPatchBudgetBuffer(
                    patch_budget=args.buffer_patch_budget,
                    seed=args.seed,
                    fold=fold,
                )
            elif args.buffer_mode == 'ratio_with_global_cap':
                buffer = GlobalPatchBudgetBuffer(
                    ratio=args.buffer_slide_ratio, seed=args.seed, fold=fold,
                    floor=args.buffer_ratio_floor, tolerance=args.buffer_budget_tolerance,
                    allow_capacity_limited=args.buffer_capacity_policy == 'capacity_limited',
                )
                if args.buffer_capacity_policy not in {'error', 'capacity_limited'}:
                    raise ValueError(f'unknown buffer_capacity_policy: {args.buffer_capacity_policy}')
            elif args.buffer_mode == 'adaptive_global_patch':
                if args.distill_method != 'kem_v4':
                    raise ValueError('adaptive_global_patch requires distill_method=kem_v4')
                if args.buffer_allocation_policy == 'unified_gradient_coverage':
                    buffer = UnifiedGradientCoverageBuffer(
                        patch_budget=args.buffer_patch_budget, seed=args.seed, fold=fold,
                        levels=_adaptive_levels(args), tolerance=0.0,
                        n_classes=args.n_classes,
                        projection_dim=args.unified_projection_dim,
                        assignment_weighted_replay=args.unified_assignment_weighted_replay,
                        objective=args.unified_objective,
                        worst_class_lexicographic=args.unified_worst_class_lexicographic,
                        trajectory_mode=args.unified_trajectory_mode,
                        assignment_view=args.unified_assignment_view,
                        class_conditioned=args.unified_class_conditioned,
                        fixed_depth64=args.unified_fixed_depth64,
                    )
                else:
                    buffer = AdaptiveGlobalPatchBuffer(
                        patch_budget=args.buffer_patch_budget, seed=args.seed, fold=fold,
                        levels=_adaptive_levels(args), tolerance=args.buffer_budget_tolerance,
                    )
            elif args.buffer_mode != 'legacy':
                raise ValueError(f'unknown buffer_mode: {args.buffer_mode}')
            elif args.buffer_balance:
                buffer = ClsssIncrementalBuffer(buffer_size=args.buffer_size)
            else:
                buffer = SimpleBuffer(buffer_size=args.buffer_size)

    results = []
    validation_results = []
    seen_classes = np.empty(0, dtype=int)
    for task in range(args.n_tasks):
        # init
        task_started = time.perf_counter()
        task_replay_updates = 0
        task_current_updates = 0
        epoch_protocol_rows = []
        epochs_completed = 0
        task_cpu_peak_start_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        torch.cuda.reset_peak_memory_stats()
        if 'old_model' in locals():
            del old_model
        torch.cuda.empty_cache()
        metrics_path = _experiment_log_path(
            args, f'fold_{fold}_task_{task}', 'metrics.csv'
        )
        if metrics_path.exists():
            metrics_path.unlink()
        logger = CSVLogger(
            root_dir=_run_log_root(args),
            name=args.exp_name,
            version=f'fold_{fold}_task_{task}',
        )

        # data
        datamodule = datamodule_gen(args, fold=fold, task=task)
        cur_classes = np.asarray(np.unique(datamodule['train_dataset'].targets), dtype=int)
        old_seen_classes = seen_classes
        seen_classes = np.append(seen_classes, cur_classes)
        unseen_classes = list(set(np.arange(args.n_classes)) - set(seen_classes))

        # optimizer
        if isinstance(args.lr, list):
            lr = float(args.lr[task])
        else:
            lr = float(args.lr)
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

        # model, optimizer = fabric.setup(model, optimizer)
        # train_loader, val_loader = fabric.setup_dataloaders(datamodule['train_loader'], datamodule['val_loader'])
        train_loader, val_loader = datamodule['train_loader'], datamodule['val_loader']

        # continual learning task setup
        if hasattr(args, 'cl_method') and args.cl_method != 'joint':
            if args.cl_method in ['LwF', 'MICIL', 'prev'] and task > 0:
                old_model = copy.deepcopy(model)
                for param in old_model.parameters():
                    param.requires_grad = False
                old_model.eval()
                old_model = fabric.to_device(old_model)
            if args.wn and task > 0:
                with torch.no_grad():
                    if args.net == 'transmil':
                        model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                    elif args.net in ['mambamil', 'chief_cached']:
                        model.classifier.weight.data = F.normalize(model.classifier.weight.data)
                    elif args.net in ['clam_sb', 'clam_mb']:
                        model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                    elif args.net == 'acmil':
                        normalize_acmil_heads(model)
                    elif args.net == 'wikg':
                        model.fc.weight.data = F.normalize(model.fc.weight.data)
                    else:
                        raise NotImplementedError
        
        # Fitting
        if isinstance(args.epochs, list):
            epochs = args.epochs[task]
        else:
            epochs = args.epochs
        early_stop_enabled = bool(args.early_stop and not args.fixed_epoch_training)
        if early_stop_enabled:
            early_stop = EarlyStopping(patience=args.patience, stop_epoch=epochs//2)
        for i in range(epochs):
            epoch_started = time.perf_counter()
            epoch_current_updates = 0
            epoch_replay_updates = 0
            current_updates_expected = _effective_loader_updates(args, train_loader)
            fixed_replay_epoch = None
            if (
                task > 0 and hasattr(args, 'cl_method')
                and hasattr(args, 'buffer_size') and args.buffer_size > 0
                and args.replay_schedule == 'fixed'
            ):
                fixed_replay_epoch = FixedReplayEpoch(
                    buffer, int(args.replay_updates_per_epoch),
                    seed=args.seed, fold=fold, task=task, epoch=i,
                    assignment_weighted=_assignment_weighted_fixed_replay(args),
                )
            pseudobag_epoch = None
            if str(args.replay_pseudobag_train) != 'none' and fixed_replay_epoch is not None:
                pseudobag_epoch = PseudoBagReplayEpoch(
                    args, buffer, fold=fold, task=task, epoch=i,
                    log_path=_experiment_log_path(args, f'fold_{fold}_task_{task}', 'pseudobag_replay.jsonl'),
                )
            # init logging
            train_loss_metric = MeanMetric()
            val_loss_metric = MeanMetric()
            if args.n_classes == 2:
                val_acc_metric = fabric.to_device(Accuracy(task='binary'))
                # val_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                val_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
                # val_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break
                
                logger_batch = {'epoch': i, 'batch': batch_idx}

                optimizer.zero_grad()
                batch = fabric.to_device(batch)

                if args.net in ['mambamil', 'chief_cached']:
                    out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100
                    loss = F.cross_entropy(out['logits'], batch['label'])

                    if hasattr(args, 'cl_method') and args.cl_on_current and task > 0:
                        old_out = old_model(batch['features'])
                        if args.cl_method == 'LwF':
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            if args.net == 'chief_cached':
                                feature_loss = F.mse_loss(out['features'], old_out['features'])
                                loss = loss + kd_loss + feature_loss
                                logger_batch.update({'kd_loss': kd_loss.item(), 'feature_loss': feature_loss.item()})
                            else:
                                loss = loss + kd_loss
                                logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL':
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            feature_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + feature_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'feature_loss': feature_loss.item()})
                        elif args.cl_method == 'prev':
                            logits_loss = 10 * kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            feature_loss = F.mse_loss(out['features'], old_out['features'])
                            attention_loss = F.kl_div(
                                torch.log(out['A'].clamp_min(1e-8)), old_out['A'], reduction='batchmean'
                            )
                            loss = loss + logits_loss + feature_loss + attention_loss
                            logger_batch.update({'logits_loss': logits_loss.item(), 'feature_loss': feature_loss.item(), 'attention_loss': attention_loss.item()})

                    train_loss_metric.update(loss.item())
                    logger_batch.update({'loss': loss.item()})
                    logger.log_metrics(logger_batch)

                elif args.net == 'wikg':
                    out = wikg_continual_forward(model, batch['features'], seen_classes)
                    loss, parts = wikg_supervised_loss(out, batch['label'])
                    train_loss_metric.update(loss.item())
                    logger_batch.update({name: value.item() for name, value in parts.items()})
                    logger_batch['loss'] = loss.item()
                    logger.log_metrics(logger_batch)

                elif args.net == 'acmil':
                    out = model(batch['features'], seen_classes=seen_classes)
                    loss, parts = acmil_supervised_loss(out, batch['label'])
                    logger_batch.update({name: value.item() for name, value in parts.items()})
                    if args.cl_on_current and task > 0:
                        if args.cl_method == 'LwF':
                            with torch.no_grad():
                                old_out = old_model(
                                    batch['features'], seen_classes=old_seen_classes
                                )
                            kd_loss = kd_loss_fn(
                                out['logits'][:, old_seen_classes],
                                old_out['logits'][:, old_seen_classes],
                            )
                            loss = loss + kd_loss
                            logger_batch['kd_loss'] = kd_loss.item()
                        elif args.cl_method == 'prev':
                            with torch.no_grad():
                                old_out = old_model(
                                    batch['features'], seen_classes=old_seen_classes
                                )
                            distill_loss, distill_parts = acmil_prev_distillation_loss(
                                out, old_out, old_seen_classes
                            )
                            loss = loss + distill_loss
                            logger_batch.update({
                                'logits_loss': distill_parts['logits_loss'].item(),
                                'attention_loss': distill_parts['attention_loss'].item(),
                                'token_loss': distill_parts['hidden_loss'].item(),
                            })
                    train_loss_metric.update(loss.item())
                    logger_batch['loss'] = loss.item()
                    logger.log_metrics(logger_batch)

                elif args.net == 'transmil':
                    out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100
                    loss = F.cross_entropy(out['logits'], batch['label'])

                    if hasattr(args, 'cl_method') and args.cl_on_current:
                        if args.cl_method == 'LwF' and task > 0:
                            old_out = old_model(batch['features'])
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            loss = loss + kd_loss
                            logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL' and task > 0:
                            old_out = old_model(batch['features'])
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + em_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                        elif args.cl_method == 'prev' and task > 0:
                            old_out = old_model(batch['features'], return_attn=True)
                            # logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            attn_loss = kd_loss_fn(out['attn1'], old_out['attn1']) + kd_loss_fn(out['attn2'], old_out['attn2']) + \
                                        F.mse_loss(out['h1'], old_out['h1']) + F.mse_loss(out['h2'], old_out['h2'])
                            # em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + attn_loss
                            logger_batch.update({'attn_loss': attn_loss.item()})


                    train_loss_metric.update(loss.item())
                    logger_batch.update({'loss': loss.item()})
                    logger.log_metrics(logger_batch)

                elif args.net in ['clam_sb', 'clam_mb']:
                    out = model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)
                    bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    inst_loss = out['instance_loss']
                    loss = 0.7*bag_loss + 0.3*inst_loss

                    if hasattr(args, 'cl_method') and args.cl_on_current:
                        if args.cl_method == 'LwF' and task > 0:
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, seen_classes=old_seen_classes)
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            loss = loss + kd_loss
                            logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL' and task > 0:
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + em_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                        elif args.cl_method == 'prev' and task > 0:
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                            # logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            attn_loss = kd_loss_fn(out['A'], old_out['A'])
                            # em_loss = F.mse_loss(out['features'], old_out['z'])
                            loss = loss + attn_loss
                            logger_batch.update({'attn_loss': attn_loss.item()})

                    train_loss_metric.update(loss.item())
                    logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item(), 'loss': loss.item()})
                    logger.log_metrics(logger_batch)
                else:
                    raise NotImplementedError

                fabric.backward(loss)
                if batch_idx % max(1, args.log_every_n_steps) == 0:
                    observability.log_train_step(
                        logger_batch, model, optimizer, task, i, batch_idx, replay=False
                    )
                optimizer.step()
                task_current_updates += 1
                epoch_current_updates += 1

                if args.wn and task > 0 and hasattr(args, 'cl_method'):
                    with torch.no_grad():
                        if args.net == 'transmil':
                            model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                        elif args.net in ['mambamil', 'chief_cached']:
                            model.classifier.weight.data = F.normalize(model.classifier.weight.data)
                        elif args.net in ['clam_sb', 'clam_mb']:
                            model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                        elif args.net == 'acmil':
                            normalize_acmil_heads(model)
                        elif args.net == 'wikg':
                            model.fc.weight.data = F.normalize(model.fc.weight.data)
                        else:
                            raise NotImplementedError

                # buffer replay
                if task > 0 and hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0:
                    if args.replay_schedule == 'fixed':
                        replay_target = fixed_replay_target(
                            epoch_current_updates - 1,
                            current_updates_expected,
                            int(args.replay_updates_per_epoch),
                        )
                        should_replay = lambda: epoch_replay_updates < replay_target
                    elif batch_idx == 0:
                        buffer.start_epoch()
                        if _conslide_buro_mode(args):
                            # Algorithm 1 of ConSlide replays one reorganized
                            # old WSI for each current WSI update.
                            should_replay = lambda: buffer.get_samples_output_count() < batch_idx + 1
                        elif _global_patch_mode(args) or _full_wsi_mode(args):
                            replay_target = int(np.ceil((batch_idx + 1) * len(buffer) / len(train_loader)))
                            should_replay = lambda: buffer.get_samples_output_count() < replay_target
                        else:
                            should_replay = lambda: batch_idx / len(train_loader) >= buffer.get_samples_output_count() / len(buffer)
                    elif _conslide_buro_mode(args):
                        # Algorithm 1 of ConSlide replays one reorganized old
                        # WSI for each current WSI update.
                        should_replay = lambda: buffer.get_samples_output_count() < batch_idx + 1
                    elif _global_patch_mode(args) or _full_wsi_mode(args):
                        replay_target = int(np.ceil((batch_idx + 1) * len(buffer) / len(train_loader)))
                        should_replay = lambda: buffer.get_samples_output_count() < replay_target
                    else:
                        should_replay = lambda: batch_idx / len(train_loader) >= buffer.get_samples_output_count() / len(buffer)
                    while should_replay():
                        old_batch = (
                            fixed_replay_epoch.next_batch()
                            if fixed_replay_epoch is not None
                            else buffer.get_next_batch()
                        )
                        if old_batch is None: raise StopIteration
                        old_batch = old_batch[0]
                        logger_batch = {'epoch': i, 'batch': batch_idx}
                        old_batch = fabric.to_device(old_batch)
                        optimizer.zero_grad()

                        if pseudobag_epoch is not None:
                            loss = pseudobag_epoch.loss(
                                old_batch,
                                lambda feats, labels: pseudobag_term_loss(
                                    args, model, feats, labels, seen_classes, unseen_classes),
                            )
                            train_loss_metric.update(loss.item())
                        elif args.net in ['mambamil', 'chief_cached']:
                            out = model(old_batch['features'])
                            out['logits'][:, unseen_classes] = -100
                            loss = F.cross_entropy(out['logits'], old_batch['label'])
                            if args.cl_method == 'LwF':
                                old_out = old_model(old_batch['features'])
                                kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                if args.net == 'chief_cached':
                                    feature_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + kd_loss + feature_loss
                                    logger_batch.update({'logits_loss': kd_loss.item(), 'feature_loss': feature_loss.item()})
                                else:
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                            elif args.cl_method == 'derpp':
                                kd_loss = stored_logit_loss(
                                    out['logits'], old_batch, old_seen_classes,
                                    args.stored_logits_lambda,
                                )
                                loss = loss + kd_loss
                                logger_batch.update({'logits_loss': kd_loss.item()})
                            elif args.cl_method == 'MICIL':
                                old_out = old_model(old_batch['features'])
                                kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                feature_loss = F.mse_loss(out['features'], old_out['features'])
                                loss = loss + 10 * kd_loss + feature_loss
                                logger_batch.update({'logits_loss': 10 * kd_loss.item(), 'feature_loss': feature_loss.item()})
                            elif args.cl_method == 'prev':
                                old_out = old_model(old_batch['features'])
                                logits_loss = 10 * kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                feature_loss = F.mse_loss(out['features'], old_out['features'])
                                attention_loss = F.kl_div(
                                    torch.log(out['A'].clamp_min(1e-8)), old_out['A'], reduction='batchmean'
                                )
                                loss = loss + logits_loss + feature_loss + attention_loss
                                logger_batch.update({'logits_loss': logits_loss.item(), 'feature_loss': feature_loss.item(), 'attention_loss': attention_loss.item()})
                            elif args.cl_method == 'ER':
                                pass
                            else:
                                raise NotImplementedError
                            train_loss_metric.update(loss.item())
                        elif args.net == 'wikg':
                            out = wikg_continual_forward(
                                model, old_batch['features'], seen_classes
                            )
                            loss, parts = wikg_supervised_loss(out, old_batch['label'])
                            logger_batch.update({
                                name: value.item() for name, value in parts.items()
                            })
                            if args.cl_method == 'derpp':
                                kd_loss = stored_logit_loss(
                                    out['logits'], old_batch, old_seen_classes,
                                    args.stored_logits_lambda,
                                )
                                loss = loss + kd_loss
                                logger_batch['logits_loss'] = kd_loss.item()
                            elif args.cl_method != 'ER':
                                raise ValueError(
                                    f"unsupported WiKG replay method {args.cl_method!r}"
                                )
                            train_loss_metric.update(loss.item())
                        elif args.net == 'acmil':
                            out = model(old_batch['features'], seen_classes=seen_classes)
                            loss, parts = acmil_supervised_loss(out, old_batch['label'])
                            logger_batch.update({name: value.item() for name, value in parts.items()})
                            if args.cl_method == 'derpp':
                                kd_loss = stored_logit_loss(
                                    out['logits'], old_batch, old_seen_classes,
                                    args.stored_logits_lambda,
                                )
                                loss = loss + kd_loss
                                logger_batch['logits_loss'] = kd_loss.item()
                            elif args.cl_method == 'LwF':
                                with torch.no_grad():
                                    old_out = old_model(
                                        old_batch['features'], seen_classes=old_seen_classes
                                    )
                                kd_loss = kd_loss_fn(
                                    out['logits'][:, old_seen_classes],
                                    old_out['logits'][:, old_seen_classes],
                                )
                                loss = loss + kd_loss
                                logger_batch['logits_loss'] = kd_loss.item()
                            elif args.cl_method == 'prev':
                                with torch.no_grad():
                                    old_out = old_model(
                                        old_batch['features'], seen_classes=old_seen_classes
                                    )
                                distill_loss, distill_parts = acmil_prev_distillation_loss(
                                    out, old_out, old_seen_classes
                                )
                                loss = loss + distill_loss
                                logger_batch.update({
                                    'logits_loss': distill_parts['logits_loss'].item(),
                                    'attention_loss': distill_parts['attention_loss'].item(),
                                    'token_loss': distill_parts['hidden_loss'].item(),
                                })
                            elif args.cl_method != 'ER':
                                raise NotImplementedError
                            train_loss_metric.update(loss.item())

                        elif args.net == 'transmil':
                            out = model(old_batch['features'], return_attn=True)
                            out['logits'][:, unseen_classes] = -100
                            loss = F.cross_entropy(out['logits'], old_batch['label'])

                            if hasattr(args, 'cl_method'):
                                if args.cl_method == 'LwF' and task > 0:
                                    old_out = old_model(old_batch['features'])
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'derpp' and task > 0:
                                    kd_loss = stored_logit_loss(
                                        out['logits'], old_batch, old_seen_classes,
                                        args.stored_logits_lambda,
                                    )
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'MICIL' and task > 0:
                                    old_out = old_model(old_batch['features'])
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    em_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + 10 * kd_loss + em_loss
                                    logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                                elif args.cl_method == 'prev' and task > 0:
                                    old_out = old_model(old_batch['features'], return_attn=True)
                                    current = dict(out)
                                    previous = dict(old_out)
                                    current['logits'] = current['logits'][:, old_seen_classes]
                                    previous['logits'] = previous['logits'][:, old_seen_classes]
                                    distill_loss, parts = replay_distillation_loss('transmil', current, previous)
                                    loss = loss + distill_loss
                                    logger_batch.update({
                                        'logits_loss': parts['logits_loss'].item(),
                                        'attn_loss': parts['attention_loss'].item(),
                                        'h_loss': parts['hidden_loss'].item(),
                                    })
                                elif args.cl_method == 'ER':
                                    pass
                                else:
                                    raise NotImplementedError

                            train_loss_metric.update(loss.item())
                        elif args.net in ['clam_sb', 'clam_mb']:
                            out = model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)
                            bag_loss = F.cross_entropy(out['logits'], old_batch['label'])
                            inst_loss = out['instance_loss']
                            loss = 0.7*bag_loss + 0.3*inst_loss
                            logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item()})

                            if hasattr(args, 'cl_method'):
                                if args.cl_method == 'LwF' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'derpp' and task > 0:
                                    kd_loss = stored_logit_loss(
                                        out['logits'], old_batch, old_seen_classes,
                                        args.stored_logits_lambda,
                                    )
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'MICIL' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    em_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + 10 * kd_loss + em_loss
                                    logger_batch.update({'logits_loss': 10 * kd_loss.item(), 'features_loss': em_loss.item()})
                                elif args.cl_method == 'prev' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    current = dict(out)
                                    previous = dict(old_out)
                                    current['logits'] = current['logits'][:, old_seen_classes]
                                    previous['logits'] = previous['logits'][:, old_seen_classes]
                                    distill_loss, parts = replay_distillation_loss(args.net, current, previous)
                                    loss = loss + distill_loss
                                    logger_batch.update({
                                        'attn_loss': parts['attention_loss'].item(),
                                        'logits_loss': parts['logits_loss'].item(),
                                    })
                                elif args.cl_method in {'ER', 'conslide_buro'}:
                                    pass
                                else:
                                    raise NotImplementedError

                            train_loss_metric.update(loss.item())
                        else:
                            raise NotImplementedError

                        if (args.buffer_mode == 'adaptive_global_patch'
                                and 'stored_embedding' in old_batch
                                and args.buffer_embedding_lambda > 0):
                            current_embedding = out['features'].reshape(1, -1)
                            stored_embedding = old_batch['stored_embedding'].reshape(1, -1)
                            embedding_loss = args.buffer_embedding_lambda * (
                                1 - F.cosine_similarity(
                                    current_embedding, stored_embedding, dim=1, eps=1e-8
                                ).mean()
                            )
                            loss = loss + embedding_loss
                            logger_batch['embedding_loss'] = embedding_loss.item()
                        if (args.buffer_mode == 'adaptive_global_patch'
                                and 'region_teacher_probs' in old_batch
                                and args.buffer_region_lambda > 0):
                            region_loss = args.buffer_region_lambda * (
                                replay_selector.kem_v4.region_replay_loss(
                                    model, old_batch['features'], old_batch
                                )
                            )
                            loss = loss + region_loss
                            logger_batch['region_replay_loss'] = region_loss.item()
                        
                        logger_batch.update({'loss': loss.item()})
                        logger.log_metrics(logger_batch)

                        fabric.backward(loss)
                        if batch_idx % max(1, args.log_every_n_steps) == 0:
                            observability.log_train_step(
                                logger_batch, model, optimizer, task, i, batch_idx, replay=True
                            )
                        optimizer.step()
                        task_replay_updates += 1
                        epoch_replay_updates += 1

                        if args.wn and task > 0 and hasattr(args, 'cl_method'):
                            with torch.no_grad():
                                if args.net == 'transmil':
                                    model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                                elif args.net == 'mambamil':
                                    model.classifier.weight.data = F.normalize(model.classifier.weight.data)
                                elif args.net in ['clam_sb', 'clam_mb']:
                                    model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                                elif args.net == 'acmil':
                                    normalize_acmil_heads(model)
                                elif args.net == 'wikg':
                                    model.fc.weight.data = F.normalize(model.fc.weight.data)
                                else:
                                    raise NotImplementedError
                                    

            # validation
            model.eval()
            val_labels, val_predictions, val_probabilities = [], [], []
            for batch_idx, batch in enumerate(val_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                batch = fabric.to_device(batch)
                if args.net in ['clam_sb', 'clam_mb']:
                    with torch.no_grad():
                        out = model(batch['features'], batch['label'], instance_eval=True, seen_classes=seen_classes)
                    bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    inst_loss = out['instance_loss']
                    loss = 0.7*bag_loss + 0.3*inst_loss
                    y_prob = F.softmax(out['logits'], dim=-1)
                    y_hat = out['logits'].argmax(dim=-1)

                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                    # val_auc_metric.update(y_prob, batch['label'])
                elif args.net == 'acmil':
                    with torch.no_grad():
                        out = model(batch['features'], seen_classes=seen_classes)
                        loss, _ = acmil_supervised_loss(out, batch['label'])
                    y_prob = F.softmax(out['logits'], dim=-1)
                    y_hat = out['logits'].argmax(dim=-1)
                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                elif args.net in ['transmil', 'mambamil', 'chief_cached']:
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100
                    loss = F.cross_entropy(out['logits'], batch['label'])
                    y_prob = F.softmax(out['logits'], dim=-1)
                    y_hat = out['logits'].argmax(dim=-1)

                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                    # val_auc_metric.update(y_prob, batch['label'])
                elif args.net == 'wikg':
                    with torch.no_grad():
                        out = wikg_continual_forward(model, batch['features'], seen_classes)
                        loss, _ = wikg_supervised_loss(out, batch['label'])
                    y_prob = out['probabilities']
                    y_hat = out['predictions']
                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                else:
                    raise NotImplementedError
                val_labels.extend(batch['label'].detach().cpu().numpy().tolist())
                val_predictions.extend(y_hat.detach().cpu().numpy().tolist())
                val_probabilities.append(y_prob.detach().cpu().numpy())
                

            # logging each epoch
            if args.net in ['clam_sb', 'clam_mb', 'acmil', 'transmil', 'mambamil', 'chief_cached', 'wikg']:
                train_loss = train_loss_metric.compute().item()
                val_loss = val_loss_metric.compute().item()
                val_acc = val_acc_metric.compute().item()
                val_auroc, val_f1 = compute_task_metrics(
                    val_labels,
                    val_predictions,
                    np.concatenate(val_probabilities, axis=0),
                    cur_classes,
                )
                # val_auc = val_auc_metric.compute().item()
                print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}\n')

                train_loss_metric.reset()
                val_loss_metric.reset()
                val_acc_metric.reset()
                # val_auc_metric.reset()
                logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss, 'val_acc_epoch': val_acc,
                                    'val_auroc_epoch': val_auroc, 'val_f1_epoch': val_f1})
                observability.log_scalars({
                    'epoch/train_loss': train_loss, 'epoch/val_loss': val_loss,
                    'epoch/val_accuracy': val_acc, 'epoch/val_macro_auroc': val_auroc,
                    'epoch/val_macro_f1': val_f1,
                    'epoch/duration_seconds': time.perf_counter() - epoch_started,
                })
                observability.log_histograms(model, task, i)
            else:
                train_loss = train_loss_metric.compute().item()
                val_loss = val_loss_metric.compute().item()
                print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n')

                train_loss_metric.reset()
                val_loss_metric.reset()
                logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss})

            if epoch_current_updates != current_updates_expected:
                raise RuntimeError(
                    f'current-update count mismatch at task={task}, epoch={i}: '
                    f'{epoch_current_updates} != {current_updates_expected}'
                )
            expected_epoch_replay = (
                int(args.replay_updates_per_epoch)
                if task > 0 and hasattr(args, 'buffer_size') and args.buffer_size > 0
                and args.replay_schedule == 'fixed' else epoch_replay_updates
            )
            if args.replay_schedule == 'fixed' and epoch_replay_updates != expected_epoch_replay:
                raise RuntimeError(
                    f'fixed replay count mismatch at task={task}, epoch={i}: '
                    f'{epoch_replay_updates} != {expected_epoch_replay}'
                )
            schedule_summary = (
                fixed_replay_epoch.summary() if fixed_replay_epoch is not None else {
                    'mode': 'natural', 'requested_updates': epoch_replay_updates,
                    'completed_updates': epoch_replay_updates,
                }
            )
            epoch_row = {
                'fold': int(fold), 'task': int(task), 'epoch': int(i),
                'current_updates': int(epoch_current_updates),
                'replay_updates': int(epoch_replay_updates),
                'replay_schedule': args.replay_schedule,
                'sampling_mode': schedule_summary['mode'],
                'replacement': bool(schedule_summary.get('replacement', False)),
                'unique_replay_items': int(schedule_summary.get('unique_replay_items', 0)),
                'replay_class_histogram': schedule_summary.get('class_histogram', {}),
                'replay_slide_histogram': schedule_summary.get('slide_histogram', {}),
            }
            epoch_protocol_rows.append(epoch_row)
            logger.log_metrics({
                'epoch': i,
                'protocol/current_updates': epoch_current_updates,
                'protocol/replay_updates': epoch_replay_updates,
            })
            observability.log_scalars({
                'protocol/current_updates': epoch_current_updates,
                'protocol/replay_updates': epoch_replay_updates,
            })
            epochs_completed = i + 1

            if early_stop_enabled:
                early_stop(i, val_loss)
                if early_stop.val_loss_min == val_loss:
                    torch.save(
                        model.state_dict(),
                        _experiment_log_path(args, f'fold_{fold}_task_{task}.pt'),
                    )
                if early_stop.early_stop:
                    print("Early stopping")
                    break

        if not early_stop_enabled:
            torch.save(
                model.state_dict(),
                _experiment_log_path(args, f'fold_{fold}_task_{task}.pt'),
            )
        training_seconds = time.perf_counter() - task_started

        replay_metrics = {
            'replay/updates': float(task_replay_updates),
            'replay/updates_per_current_sample': (
                float(task_replay_updates) / max(len(train_loader), 1)
            ),
        }
        logger.log_metrics(replay_metrics)
        observability.log_scalars(replay_metrics)

        model.load_state_dict(torch.load(
            _experiment_log_path(args, f'fold_{fold}_task_{task}.pt')
        ))
        model.eval()
        torch.cuda.empty_cache()
        buffer_update_started = time.perf_counter()
        # add data in buffer
        if (hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0
                and _conslide_buro_mode(args)):
            seed_everything(args.seed)
            plan = buffer.add_task_dataset(
                datamodule['train_dataset'], task,
                coords_loader=lambda slide_id, features: _buffer_coords(
                    args, slide_id, features
                ),
                max_slides=args.conslide_buffer_max_slides,
            )
            task_log_dir = _experiment_log_path(args, f'fold_{fold}_task_{task}')
            os.makedirs(task_log_dir, exist_ok=True)
            with open(os.path.join(task_log_dir, 'buffer_plan.json'), 'w', encoding='utf-8') as handle:
                json.dump(plan, handle, indent=2, ensure_ascii=False)
            pd.DataFrame(buffer.manifest_rows()).to_csv(
                os.path.join(task_log_dir, 'buffer_manifest.csv'), index=False,
            )
            torch.save(
                buffer.state_dict(), os.path.join(task_log_dir, 'conslide_buro_buffer.pt')
            )
            metrics = {
                'buffer_size': len(buffer),
                'n_patches_in_buffer': buffer.n_patches_total,
                'buffer/target_patches': args.buffer_patch_budget,
                'buffer/stored_regions': len(buffer.records),
                'buffer/stored_wsi_count': len(buffer),
                'buffer/replay_updates': buffer.total_replay_updates,
                'buffer/budget_underfill': plan['budget_underfill'],
            }
            logger.log_metrics(metrics)
            observability.log_scalars(metrics)
            observability.log_buffer(buffer)
            print(
                f'ConSlide-BuRo buffer: {len(buffer)} source WSI, '
                f'{len(buffer.records)} regions, {buffer.n_patches_total}/'
                f'{args.buffer_patch_budget} patches'
            )

        if (hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0
                and _global_patch_mode(args)):
            seed_everything(args.seed)
            if args.buffer_mode == 'adaptive_global_patch':
                target_budget = int(args.buffer_patch_budget)
                if args.buffer_allocation_policy == 'unified_gradient_coverage':
                    candidates, selector_metrics = _new_unified_candidates(
                        args, datamodule['train_dataset'], buffer, model, fabric,
                        seen_classes, task, replay_selector,
                    )
                else:
                    candidates, selector_metrics = _new_adaptive_candidates(
                        args, datamodule['train_dataset'], buffer, model, fabric,
                        seen_classes, task, replay_selector,
                    )
            else:
                target_budget = load_patch_budget(args.buffer_budget_file, fold, task)
                selector_metrics = {}
                candidates = _new_ratio_candidates(
                    args, datamodule['train_dataset'], buffer, fold, task,
                )
            admitted, plan = buffer.plan(candidates, target_budget, task)
            if (args.buffer_allocation_policy == 'unified_gradient_coverage'
                    and args.buffer_region_lambda > 0 and plan.get('shrunk_slide_ids')):
                shrunk_ids = set(plan['shrunk_slide_ids'])
                for stored_sample in buffer._all_samples():
                    if buffer._sample_slide_id(stored_sample) not in shrunk_ids:
                        continue
                    targets = replay_selector.kem_v4.build_region_targets(
                        model, stored_sample['features'], seen_classes, coords=None,
                    )
                    stored_sample.update({
                        name: value.to(stored_sample['features'].device)
                        for name, value in targets.items()
                    })
            selection_seconds = []
            selection_candidate_patches = []
            selected_ranks = []
            partition_coverages = []
            partition_counts = []
            for candidate in admitted:
                sample = datamodule['train_dataset'][candidate.dataset_index]
                if str(sample['slide_id']) != candidate.slide_id:
                    raise AssertionError('dataset index changed after deterministic admission')
                if int(sample['n_patch']) != candidate.original_patch_count:
                    raise AssertionError('canonical patch count differs from loaded WSI features')
                batch = _select_buffer_sample(
                    args, sample, candidate.patch_budget, replay_selector, model,
                    fabric, seen_classes, task, observability,
                    candidate=candidate,
                )
                diagnostics = batch.get('selection_diagnostics', {})
                selection_time = diagnostics.get(
                    'selector/selection_time_seconds',
                    diagnostics.get('kem_v4/selection_time_seconds'),
                )
                if selection_time is not None:
                    selection_seconds.append(float(selection_time))
                if not isinstance(candidate, SelectedReplayOption):
                    selection_candidate_patches.append(int(candidate.original_patch_count))
                if 'kem_v4/selected_effective_rank' in diagnostics:
                    selected_ranks.append(float(diagnostics['kem_v4/selected_effective_rank']))
                coverage = diagnostics.get('selector/partition_coverage')
                if coverage is not None:
                    partition_coverages.append(float(coverage))
                count = diagnostics.get('selector/partition_count')
                if count is not None:
                    partition_counts.append(float(count))
                buffer.add(batch)
            buffer.finalize_plan()
            selector_metrics.update({
                'selector/final_selection_seconds': float(sum(selection_seconds)),
                'selector/final_selection_seconds_per_wsi': (
                    float(np.mean(selection_seconds)) if selection_seconds else 0.0
                ),
                'selector/analyzed_wsi': float(len(admitted)),
                'selector/analyzed_patch_count': float(sum(selection_candidate_patches)),
                'selector/partition_coverage_mean': (
                    float(np.mean(partition_coverages)) if partition_coverages else np.nan
                ),
                'selector/partition_count_mean': (
                    float(np.mean(partition_counts)) if partition_counts else np.nan
                ),
                'kem_v4/final_selection_seconds': float(sum(selection_seconds)),
                'kem_v4/final_selection_seconds_per_wsi': (
                    float(np.mean(selection_seconds)) if selection_seconds else 0.0
                ),
                'kem_v4/selector_total_seconds': (
                    float(selector_metrics.get(
                        'kem_v4/admission_analysis_seconds',
                        selector_metrics.get('unified/option_analysis_seconds', 0.0),
                    ))
                    + float(sum(selection_seconds))
                ),
                'kem_v4/selected_effective_rank_mean': (
                    float(np.mean(selected_ranks)) if selected_ranks else np.nan
                ),
            })
            if args.buffer_allocation_policy == 'unified_gradient_coverage':
                selector_metrics.update({
                    'unified/coverage': float(buffer.last_plan['coverage']),
                    'unified/quantization_error': float(buffer.last_plan['quantization_error']),
                    'unified/shrunk_slides': float(buffer.last_plan['shrunk_slides']),
                    'unified/selection_objective_value': float(
                        buffer.last_plan['selection_objective_value']
                    ),
                    'unified/selector_total_seconds': (
                        float(selector_metrics.get('unified/option_analysis_seconds', 0.0))
                        + float(sum(selection_seconds))
                    ),
                })
            buffer.last_plan['selector_metrics'] = selector_metrics
            actual_ids = sorted(item.slide_id for item in buffer.existing_candidates())
            if actual_ids != plan['selected_slide_ids']:
                raise AssertionError('actual buffer WSI set differs from deterministic admission plan')
            task_log_dir = _experiment_log_path(args, f'fold_{fold}_task_{task}')
            os.makedirs(task_log_dir, exist_ok=True)
            with open(os.path.join(task_log_dir, 'buffer_plan.json'), 'w', encoding='utf-8') as handle:
                json.dump(buffer.last_plan, handle, indent=2, ensure_ascii=False)
            manifest_rows = []
            for label, samples in sorted(buffer.buffers.items()):
                for sample in samples:
                    slide_id = buffer._sample_slide_id(sample)
                    original = int(torch.as_tensor(sample['original_patch_count']).item())
                    selected = int(sample['features'].shape[0])
                    manifest_rows.append({
                        'slide_id': slide_id, 'label': int(label),
                        'source_task': int(torch.as_tensor(sample['source_task']).item()),
                        'original_patch_count': original, 'selected_patch_count': selected,
                        'requested_ratio': args.buffer_slide_ratio,
                        'actual_ratio': selected / original,
                        'floor_hit': int(torch.as_tensor(sample['ratio_floor_hit']).item()),
                        'small_bag': int(torch.as_tensor(sample['ratio_small_bag']).item()),
                        'selection_method': str(sample.get('selection_method', args.distill_method)),
                        'buffer_mode': args.buffer_mode,
                        'pseudo_bag_count': int(torch.unique(
                            torch.as_tensor(sample.get('pseudo_bag_ids', torch.empty(0, dtype=torch.long)))
                        ).numel()),
                        'partition_coverage': float(sample.get(
                            'selection_diagnostics', {}
                        ).get('selector/partition_coverage', np.nan)),
                        'allocation_utility': float(torch.as_tensor(
                            sample.get('allocation_utility', 0.0)
                        ).item()),
                    })
            pd.DataFrame(manifest_rows).to_csv(
                os.path.join(task_log_dir, 'buffer_manifest.csv'), index=False,
            )
            buffer_metrics = {
                'buffer_size': len(buffer), 'n_patches_in_buffer': buffer.n_patches_total,
                'buffer/target_patches': target_budget,
                'buffer/effective_target_patches': buffer.last_plan['target_effective'],
                'buffer/budget_error': buffer.last_plan['actual_budget_error'],
                'buffer/original_underfill_ratio': buffer.last_plan['original_underfill_ratio'],
                'buffer/capacity_limited': float(buffer.last_plan['capacity_limited']),
                'buffer/admitted_slides': buffer.last_plan['admitted_slides'],
                'buffer/evicted_slides': buffer.last_plan['evicted_slides'],
                'buffer/floor_hits': buffer.last_plan['floor_hits'],
                'buffer/small_bags': buffer.last_plan['small_bags'],
                **selector_metrics,
            }
            logger.log_metrics(buffer_metrics)
            observability.log_scalars(buffer_metrics)
            observability.log_buffer(buffer)
            print(
                f'Global patch buffer: {len(buffer)} WSI, {buffer.n_patches_total}/'
                f'{buffer.last_plan["target_effective"]} effective patches '
                f'(original target {target_budget}, capacity_limited={buffer.last_plan["capacity_limited"]})'
            )
            print(f'Labels in buffer: {buffer.labels}')

            if args.head_calibration and task > 0:
                before = evaluate_continual_loaders(
                    model, validation_dataloaders, args, fabric, seen_classes,
                    unseen_classes, task,
                )
                calibration = calibrate_classification_head(
                    model, buffer, args, fabric, seen_classes, task,
                )
                after = evaluate_continual_loaders(
                    model, validation_dataloaders, args, fabric, seen_classes,
                    unseen_classes, task,
                )
                calibration.update({f'head_calibration/before_{key}': value for key, value in before.items()})
                calibration.update({f'head_calibration/after_{key}': value for key, value in after.items()})
                observability.log_scalars(calibration)
                logger.log_metrics(calibration)
                torch.save(
                    model.state_dict(),
                    _experiment_log_path(args, f'fold_{fold}_task_{task}.pt'),
                )

            if args.save_buffer_snapshots:
                snapshot_config = {
                    'seed': args.seed, 'cl_method': args.cl_method,
                    'model': args.net, 'distill_method': args.distill_method,
                    'selection_method': args.distill_method,
                    'buffer_size': args.buffer_size,
                    'buffer_slide_size': args.buffer_slide_size,
                    'kem_anchor_method': args.kem_anchor_method,
                    'kem_anchor_ratio': args.kem_anchor_ratio,
                    'stored_logits_lambda': args.stored_logits_lambda,
                    'head_calibration': args.head_calibration,
                    'buffer_mode': args.buffer_mode,
                    'buffer_patch_budget': args.buffer_patch_budget,
                    'buffer_patch_levels': args.buffer_patch_levels,
                    'buffer_allocation_policy': args.buffer_allocation_policy,
                    'buffer_patch_selector': args.buffer_patch_selector,
                    'buffer_embedding_lambda': args.buffer_embedding_lambda,
                    'buffer_region_lambda': args.buffer_region_lambda,
                    'unified_objective': args.unified_objective,
                    'unified_worst_class_lexicographic': args.unified_worst_class_lexicographic,
                    'unified_region_stratified_candidates': args.unified_region_stratified_candidates,
                    'unified_assignment_weighted_replay': args.unified_assignment_weighted_replay,
                }
                snapshot_dir = save_buffer_snapshot(
                    buffer, args.buffer_snapshot_dir, args.exp_name, fold, task,
                    snapshot_config, args.buffer_snapshot_fp16,
                    args.buffer_snapshot_full,
                )
                observability.log_scalars({'buffer/snapshot_task': float(task)})
                print(f'Buffer snapshot: {snapshot_dir}')

        if (hasattr(args, 'cl_method') and hasattr(args, 'buffer_size')
                and args.buffer_size > 0 and _full_wsi_mode(args)):
            seed_everything(args.seed)

            def prepare_full_wsi(sample):
                if args.cl_method != 'derpp':
                    return sample
                with torch.no_grad():
                    device_sample = fabric.to_device(sample)
                    output = ReplayModelAdapter(args.net).forward(
                        model, device_sample['features'], seen_classes
                    )
                sample['logits'] = output.logits.detach().clone()
                sample['stored_embedding'] = output.features.detach().clone()
                sample['stored_classes'] = torch.as_tensor(
                    seen_classes, device=output.logits.device, dtype=torch.long
                )
                return sample

            plan = buffer.update(
                datamodule['train_dataset'], task, transform=prepare_full_wsi
            )
            task_log_dir = _experiment_path(args, f'fold_{fold}_task_{task}')
            os.makedirs(task_log_dir, exist_ok=True)
            with open(os.path.join(
                task_log_dir, 'buffer_plan.json'
            ), 'w', encoding='utf-8') as handle:
                json.dump(plan, handle, indent=2, ensure_ascii=False)
            pd.DataFrame(buffer.manifest_rows()).to_csv(
                os.path.join(task_log_dir, 'buffer_manifest.csv'), index=False,
            )
            metrics = {
                'buffer_size': len(buffer),
                'n_patches_in_buffer': buffer.n_patches_total,
                'buffer/target_patches': args.buffer_patch_budget,
                'buffer/stored_wsi_count': len(buffer),
                'buffer/budget_underfill': plan['budget_underfill'],
                'buffer/full_wsi_replay': 1.0,
            }
            logger.log_metrics(metrics)
            observability.log_scalars(metrics)
            observability.log_buffer(buffer)
            print(
                f'Full-WSI buffer: {len(buffer)} complete WSI, '
                f'{buffer.n_patches_total}/{args.buffer_patch_budget} patches'
            )

        if (hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0
                and not _global_patch_mode(args) and not _conslide_buro_mode(args)
                and not _full_wsi_mode(args)):
            seed_everything(args.seed)
            # calculete the number of samples to add in buffer
            buffer.adjust_buffer_size_by_new_classes(len(cur_classes))
            n_seen_samples_per_task = min(args.buffer_size, len(train_loader))
            # add samples in buffer
            for batch_idx, batch in enumerate(train_loader):
                # seed_everything(args.seed + batch_idx)
                buffer_idx = buffer.add_judge(batch['label'].item())
                if buffer_idx == -1: continue
                if args.buffer_slide_size > 0:
                    original_patch_count = int(batch['features'].size(0))
                    buffer_slide_size = args.buffer_slide_size * batch['features'].size(0) if isinstance(args.buffer_slide_size, float) else args.buffer_slide_size
                    batch = fabric.to_device(batch)
                    attn = None
                    with torch.no_grad():
                        if args.net in ['clam_sb', 'clam_mb']:
                            out = model(batch['features'], batch['label'], seen_classes=seen_classes)
                            attn = out['A'].mean(dim=0, keepdim=True).view(1, -1)
                        elif args.net == 'acmil':
                            out = model(batch['features'], seen_classes=seen_classes)
                            attn = out['A'].reshape(1, -1)
                        elif args.net == 'transmil':
                            if args.distill_method != 'kem':
                                out = model(batch['features'], return_attn=True)
                                epsilon = torch.finfo(out['attn1'].dtype).eps
                                attn1 = out['attn1'] / out['attn1'].max(dim=1, keepdim=True)[0].clamp_min(epsilon)
                                attn2 = out['attn2'] / out['attn2'].max(dim=1, keepdim=True)[0].clamp_min(epsilon)
                                attn = (attn2 + 1) * (attn1 + 1) / 4
                                attn = attn.mean(dim=0, keepdim=True).view(1, -1)
                        elif args.net == 'mambamil':
                            if args.distill_method == 'kem':
                                raise ValueError(
                                    'legacy soft-mask KEM is invalid for MambaMIL; use distill_method=kem_v2'
                                )
                            out = model(batch['features'])
                            attn = out['A'].mean(dim=1).view(1, -1)
                        elif args.net == 'chief_cached':
                            raise ValueError('chief_cached replay stores complete slide embeddings; set buffer_slide_size=0')
                        else:
                            raise NotImplementedError
                    selection = replay_selector.select(
                        features=batch['features'],
                        budget=buffer_slide_size,
                        scores=attn,
                        model=model,
                        label=batch['label'],
                        seen_classes=seen_classes,
                        coords=(replay_selector.kem_v2.region_supervisor.load_and_validate_coords(
                            batch['slide_id'][0], batch['features'].detach().cpu()
                        ) if args.region_supervision else None),
                        slide_id=batch['slide_id'][0],
                        task=task,
                    )
                    batch['features'] = batch['features'].index_select(
                        0, selection.indices.to(batch['features'].device)
                    )
                    batch['selected_indices'] = selection.indices.detach().clone()
                    batch['original_patch_count'] = torch.tensor(original_patch_count)
                    batch['selection_method'] = args.distill_method
                    if selection.diagnostics:
                        observability.log_scalars(selection.diagnostics)
                        batch['selection_diagnostics'] = dict(selection.diagnostics)
                        
                if args.cl_method in ['derpp']:
                    with torch.no_grad():
                        batch = fabric.to_device(batch)
                        if args.net in ['clam_sb', 'clam_mb']:
                            out = model(batch['features'], batch['label'], seen_classes=seen_classes)
                        elif args.net == 'acmil':
                            out = model(batch['features'], seen_classes=seen_classes)
                        elif args.net in ['transmil', 'mambamil', 'chief_cached']:
                            out = model(batch['features'])
                        elif args.net == 'wikg':
                            out = wikg_continual_forward(model, batch['features'], seen_classes)
                    batch['logits'] = out['logits'].detach().clone()
                    batch['stored_classes'] = torch.as_tensor(
                        seen_classes, device=out['logits'].device, dtype=torch.long
                    )
                batch['source_task'] = torch.tensor(task)
                        
                buffer.add(batch, buffer_idx)
                # if batch_idx >= n_seen_samples_per_task:
                #     break

            print(f'Buffer size: {len(buffer)}')
            print(f'Number of patches in buffer: {buffer.n_patches_total}')
            print(f'Labels in buffer: {buffer.labels}')
            # save buffer.labels to csv
            with _experiment_log_path(
                args, f'fold_{fold}_task_{task}', 'buffer_labels.csv'
            ).open('a') as f:
                for key in buffer.labels.keys():
                    f.write("%s,%s\n"%(key,buffer.labels[key]))
            logger.log_metrics({'buffer_size': len(buffer), 'n_patches_in_buffer': buffer.n_patches_total})
            observability.log_buffer(buffer)

            if args.head_calibration and task > 0:
                before = evaluate_continual_loaders(
                    model, validation_dataloaders, args, fabric, seen_classes,
                    unseen_classes, task,
                )
                calibration = calibrate_classification_head(
                    model, buffer, args, fabric, seen_classes, task,
                )
                after = evaluate_continual_loaders(
                    model, validation_dataloaders, args, fabric, seen_classes,
                    unseen_classes, task,
                )
                calibration.update({f'head_calibration/before_{key}': value for key, value in before.items()})
                calibration.update({f'head_calibration/after_{key}': value for key, value in after.items()})
                observability.log_scalars(calibration)
                logger.log_metrics(calibration)
                torch.save(
                    model.state_dict(),
                    _experiment_log_path(args, f'fold_{fold}_task_{task}.pt'),
                )

            if args.save_buffer_snapshots:
                snapshot_config = {
                    'seed': args.seed, 'cl_method': args.cl_method,
                    'model': args.net,
                    'distill_method': args.distill_method,
                    'buffer_size': args.buffer_size,
                    'buffer_slide_size': args.buffer_slide_size,
                    'kem_anchor_method': args.kem_anchor_method,
                    'kem_anchor_ratio': args.kem_anchor_ratio,
                    'stored_logits_lambda': args.stored_logits_lambda,
                    'head_calibration': args.head_calibration,
                    'buffer_mode': args.buffer_mode,
                    'buffer_patch_budget': args.buffer_patch_budget,
                    'buffer_patch_levels': args.buffer_patch_levels,
                    'buffer_allocation_policy': args.buffer_allocation_policy,
                    'buffer_patch_selector': args.buffer_patch_selector,
                    'buffer_embedding_lambda': args.buffer_embedding_lambda,
                    'buffer_region_lambda': args.buffer_region_lambda,
                    'unified_objective': args.unified_objective,
                    'unified_worst_class_lexicographic': args.unified_worst_class_lexicographic,
                    'unified_region_stratified_candidates': args.unified_region_stratified_candidates,
                    'unified_assignment_weighted_replay': args.unified_assignment_weighted_replay,
                }
                snapshot_dir = save_buffer_snapshot(
                    buffer, args.buffer_snapshot_dir, args.exp_name, fold, task,
                    snapshot_config, args.buffer_snapshot_fp16,
                    args.buffer_snapshot_full,
                )
                observability.log_scalars({'buffer/snapshot_task': float(task)})
                print(f'Buffer snapshot: {snapshot_dir}')

        _validate_strict_budget(args, buffer, task)
        buffer_update_seconds = time.perf_counter() - buffer_update_started
        task_duration_seconds = time.perf_counter() - task_started
        inventory = _buffer_inventory(buffer)
        task_log_dir = _experiment_path(args, f'fold_{fold}_task_{task}')
        os.makedirs(task_log_dir, exist_ok=True)
        protocol = {
            'schema_version': 'kem-paper-strict5k-protocol-v1',
            'fold': int(fold), 'task': int(task),
            'epochs_requested': int(epochs),
            'epochs_completed': int(epochs_completed),
            'fixed_epoch_training': bool(args.fixed_epoch_training),
            'replay_schedule': args.replay_schedule,
            'replay_updates_per_epoch': int(args.replay_updates_per_epoch),
            'current_updates': int(task_current_updates),
            'replay_updates': int(task_replay_updates),
            'training_seconds': float(training_seconds),
            'buffer_update_seconds': float(buffer_update_seconds),
            'task_duration_seconds': float(task_duration_seconds),
            **inventory,
            'strict_patch_budget': bool(args.strict_patch_budget),
            'patch_budget': int(args.buffer_patch_budget),
            'gpu_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
            'gpu_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
            'cpu_peak_rss_kb': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            'cpu_peak_rss_delta_kb': max(
                0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                - task_cpu_peak_start_kb,
            ),
            'epochs': epoch_protocol_rows,
        }
        if args.fixed_epoch_training and epochs_completed != int(epochs):
            raise RuntimeError(
                f'fixed training stopped early at task {task}: '
                f'{epochs_completed}/{epochs} epochs'
            )
        if args.replay_schedule == 'fixed':
            expected_task_replay = (
                0 if task == 0 or args.buffer_size <= 0
                else int(epochs) * int(args.replay_updates_per_epoch)
            )
            if task_replay_updates != expected_task_replay:
                raise RuntimeError(
                    f'task replay count mismatch: {task_replay_updates} '
                    f'!= {expected_task_replay}'
                )
        with open(os.path.join(
            task_log_dir, 'protocol_audit.json'
        ), 'w', encoding='utf-8') as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=False)
        pd.DataFrame(epoch_protocol_rows).to_csv(
            os.path.join(task_log_dir, 'protocol_epochs.csv'), index=False,
        )
        protocol_metrics = {
            'protocol/task_current_updates': task_current_updates,
            'protocol/task_replay_updates': task_replay_updates,
            'protocol/stored_patches': inventory['stored_patches'],
            'protocol/stored_wsi': inventory['stored_wsi'],
            'protocol/stored_pseudobags': inventory['stored_pseudobags'],
            'runtime/peak_gpu_allocated_mb': protocol['gpu_peak_allocated_bytes'] / 1024**2,
            'runtime/peak_gpu_reserved_mb': protocol['gpu_peak_reserved_bytes'] / 1024**2,
            'runtime/peak_cpu_rss_mb': protocol['cpu_peak_rss_kb'] / 1024,
        }
        logger.log_metrics(protocol_metrics)
        observability.log_scalars(protocol_metrics)

        del train_loader, val_loader, optimizer, datamodule

        # testing, log acc and auc
        model.eval()
        if args.n_classes == 2:
            test_acc_metric = fabric.to_device(Accuracy(task='binary'))
            # test_auc_metric = fabric.to_device(AUROC(task='binary'))
        else:
            test_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
            # test_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))
        result = {'fold': fold, 'task': task}
        task_confusions = {}
        for idx, test_loader in enumerate(test_dataloaders):
            task_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
            all_labels, all_preds, all_probs = [], [], []
            for batch_idx, batch in enumerate(test_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                batch = fabric.to_device(batch)

                if args.net in ['clam_sb', 'clam_mb', 'acmil']:
                    with torch.no_grad():
                        out = model(batch['features'], seen_classes=seen_classes)
                    # bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    y_hat = out['logits'].argmax(dim=-1)
                    y_prob = F.softmax(out['logits'], dim=-1)

                    test_acc_metric.update(y_hat, batch['label'])
                    # test_auc_metric.update(y_prob, batch['label'])
                elif args.net in ['transmil', 'mambamil', 'chief_cached']:
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100
                    y_hat = out['logits'].argmax(dim=-1)
                    y_prob = F.softmax(out['logits'], dim=-1)

                    test_acc_metric.update(y_hat, batch['label'])
                    # test_auc_metric.update(y_prob, batch['label'])
                elif args.net == 'wikg':
                    with torch.no_grad():
                        out = wikg_continual_forward(model, batch['features'], seen_classes)
                    y_hat = out['predictions']
                    y_prob = out['probabilities']
                    test_acc_metric.update(y_hat, batch['label'])
                else:
                    raise NotImplementedError
                all_labels.extend(batch['label'].detach().cpu().numpy().tolist())
                all_preds.extend(y_hat.detach().cpu().numpy().tolist())
                all_probs.append(y_prob.detach().cpu().numpy())

            accuracy = test_acc_metric.compute().item()
            if idx > task:
                macro_auroc, macro_f1 = np.nan, 0.0
            else:
                macro_auroc, macro_f1 = compute_task_metrics(
                    all_labels,
                    all_preds,
                    np.concatenate(all_probs, axis=0),
                    task_classes,
                )
            result.update({
                f'{idx}_acc': accuracy,
                f'{idx}_auroc': macro_auroc,
                f'{idx}_f1': macro_f1,
            })
            if idx <= task:
                task_confusions[idx] = confusion_matrix(all_labels, all_preds, labels=task_classes)

            test_acc_metric.reset()
            # test_auc_metric.reset()

        print(result)
        results.append(result)
        validation_result = {'fold': fold, 'task': task}
        validation_result.update(evaluate_continual_loaders(
            model, validation_dataloaders, args, fabric, seen_classes,
            unseen_classes, task,
        ))
        validation_results.append(validation_result)
        pd.DataFrame(validation_results).to_csv(
            _experiment_log_path(args, f'fold_{fold}_validation_results.csv'),
            index=False,
        )
        active_validation_acc = [
            validation_result[f'{idx}_acc'] for idx in range(task + 1)
        ]
        observability.log_scalars({
            'validation_continual/AACC': float(np.mean(active_validation_acc)),
            **{
                f'validation_continual/task_{idx}_accuracy': validation_result[f'{idx}_acc']
                for idx in range(task + 1)
            },
        })
        observability.log_continual(results, task, task_confusions)
        df = pd.DataFrame(results)
        df.to_csv(
            _experiment_log_path(args, f'fold_{fold}_results.csv'), index=False
        )
        
        logger.finalize(f"Success on fold {fold} task {task}!")

    observability.close()
    return results # [{'fold': 0, 'task': 0, '0_auc': 0.9, '0_acc': 0.8}, {...}, ...]


def one_fold_jt(args, fold=0):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; CPU fallback is disabled')
    if args.net == 'chief_cached' and getattr(args, 'cl_method', None) != 'joint':
        raise ValueError('one_fold_jt with chief_cached requires cl_method=joint')
    if args.net == 'wikg':
        validate_wikg_continual_config(args)
    if args.replay_schedule not in {'natural', 'fixed'}:
        raise ValueError('replay_schedule must be natural or fixed')
    torch.cuda.reset_peak_memory_stats()
    joint_cpu_peak_start_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    fabric = Fabric(devices=1, accelerator="cuda")
    seed_everything(args.seed)
    observability = ExperimentLogger(args, fold, root=_run_log_root(args))

    model = load_model(args)
    model = fabric.to_device(model)

    # Prepare test dataloaders from all tasks
    datamodule = datamodule_gen(args, fold=fold, task=args.n_tasks)

    # optimizer
    if isinstance(args.lr, list):
        lr = float(args.lr[0])
    else:
        lr = float(args.lr)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    # model, optimizer = fabric.setup(model, optimizer)
    # train_loader, val_loader = fabric.setup_dataloaders(datamodule['train_loader'], datamodule['val_loader'])
    train_loader, val_loader = datamodule['train_loader'], datamodule['val_loader']
    
    logger = CSVLogger(
        root_dir=_run_log_root(args), name=args.exp_name, version=f'fold_{fold}_JT'
    )

    # Fitting
    if isinstance(args.epochs, list):
        epochs = args.epochs[0]
    else:
        epochs = args.epochs
    early_stop_enabled = bool(args.early_stop and not args.fixed_epoch_training)
    if early_stop_enabled:
        early_stop = EarlyStopping(patience=args.patience, stop_epoch=epochs//2)

    joint_current_updates = 0
    joint_epoch_rows = []
    epochs_completed = 0
    for i in range(epochs):
        epoch_started = time.perf_counter()
        epoch_current_updates = 0
        current_updates_expected = _effective_loader_updates(args, train_loader)
        # init logging
        train_loss_metric = MeanMetric()
        val_loss_metric = MeanMetric()
        if args.n_classes == 2:
            val_acc_metric = fabric.to_device(Accuracy(task='binary'))
            # val_auc_metric = fabric.to_device(AUROC(task='binary'))
        else:
            val_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
            # val_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break
            
            logger_batch = {'epoch': i, 'batch': batch_idx}

            optimizer.zero_grad()
            batch = fabric.to_device(batch)

            if args.net in ['transmil', 'mambamil', 'chief_cached', 'wikg']:
                out = model(batch['features'])
                loss = F.cross_entropy(out['logits'], batch['label'])

                train_loss_metric.update(loss.item())
                logger_batch.update({'loss': loss.item()})
                logger.log_metrics(logger_batch)

            elif args.net == 'acmil':
                out = model(batch['features'], seen_classes=list(range(args.n_classes)))
                loss, parts = acmil_supervised_loss(out, batch['label'])
                train_loss_metric.update(loss.item())
                logger_batch.update({name: value.item() for name, value in parts.items()})
                logger_batch['loss'] = loss.item()
                logger.log_metrics(logger_batch)

            elif args.net in ['clam_sb', 'clam_mb']:
                out = model(batch['features'], batch['label'], instance_eval=True, return_features=True)
                bag_loss = F.cross_entropy(out['logits'], batch['label'])
                inst_loss = out['instance_loss']
                loss = 0.7*bag_loss + 0.3*inst_loss

                train_loss_metric.update(loss.item())
                logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item(), 'loss': loss.item()})
                logger.log_metrics(logger_batch)
            else:
                raise NotImplementedError

            fabric.backward(loss)
            if batch_idx % max(1, args.log_every_n_steps) == 0:
                observability.log_train_step(
                    logger_batch, model, optimizer, 0, i, batch_idx, replay=False
                )
            optimizer.step()
            joint_current_updates += 1
            epoch_current_updates += 1

        # validation
        model.eval()
        val_labels, val_predictions, val_probabilities = [], [], []
        for batch_idx, batch in enumerate(val_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break

            batch = fabric.to_device(batch)

            if args.net == 'acmil':
                with torch.no_grad():
                    out = model(batch['features'], seen_classes=list(range(args.n_classes)))
                    loss, _ = acmil_supervised_loss(out, batch['label'])
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)
                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
            elif args.net in ['clam_sb', 'clam_mb']:
                with torch.no_grad():
                    out = model(batch['features'], batch['label'], instance_eval=True)
                bag_loss = F.cross_entropy(out['logits'], batch['label'])
                inst_loss = out['instance_loss']
                loss = 0.7*bag_loss + 0.3*inst_loss
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)

                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
                # val_auc_metric.update(y_prob, batch['label'])
            elif args.net in ['transmil', 'mambamil', 'chief_cached', 'wikg']:
                with torch.no_grad():
                    out = model(batch['features'])
                loss = F.cross_entropy(out['logits'], batch['label'])
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)

                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
                # val_auc_metric.update(y_prob, batch['label'])
            else:
                raise NotImplementedError
            val_labels.extend(batch['label'].detach().cpu().numpy().tolist())
            val_predictions.extend(y_hat.detach().cpu().numpy().tolist())
            val_probabilities.append(y_prob.detach().cpu().numpy())
            

        # logging each epoch
        if args.net in ['clam_sb', 'clam_mb', 'transmil', 'acmil', 'mambamil', 'chief_cached', 'wikg']:
            train_loss = train_loss_metric.compute().item()
            val_loss = val_loss_metric.compute().item()
            val_acc = val_acc_metric.compute().item()
            val_classes = np.asarray(sorted(set(val_labels)), dtype=int)
            val_auroc, val_f1 = compute_task_metrics(
                val_labels, val_predictions, np.concatenate(val_probabilities), val_classes
            )
            # val_auc = val_auc_metric.compute().item()
            print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}\n')

            train_loss_metric.reset()
            val_loss_metric.reset()
            val_acc_metric.reset()
            # val_auc_metric.reset()
            logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss, 'val_acc_epoch': val_acc,
                                'val_auroc_epoch': val_auroc, 'val_f1_epoch': val_f1})
            observability.log_scalars({
                'epoch/train_loss': train_loss, 'epoch/val_loss': val_loss,
                'epoch/val_accuracy': val_acc, 'epoch/val_macro_auroc': val_auroc,
                'epoch/val_macro_f1': val_f1,
                'epoch/duration_seconds': time.perf_counter() - epoch_started,
            })
            observability.log_histograms(model, 0, i)
        else:
            train_loss = train_loss_metric.compute().item()
            val_loss = val_loss_metric.compute().item()
            print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n')

            train_loss_metric.reset()
            val_loss_metric.reset()
            logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss})

        if epoch_current_updates != current_updates_expected:
            raise RuntimeError(
                f'joint current-update mismatch at epoch {i}: '
                f'{epoch_current_updates} != {current_updates_expected}'
            )
        epochs_completed = i + 1
        joint_epoch_rows.append({
            'fold': int(fold), 'task': 'joint', 'epoch': int(i),
            'current_updates': int(epoch_current_updates), 'replay_updates': 0,
            'replay_schedule': 'none', 'sampling_mode': 'none',
            'replacement': False, 'unique_replay_items': 0,
        })
        if early_stop_enabled:
            early_stop(i, val_loss)
            if early_stop.val_loss_min == val_loss:
                torch.save(
                    model.state_dict(),
                    _experiment_log_path(args, f'fold_{fold}_JT.pt'),
                )
            if early_stop.early_stop:
                print("Early stopping")
                break

    if not early_stop_enabled:
        torch.save(model.state_dict(), _experiment_path(args, f'fold_{fold}_JT.pt'))
    if args.fixed_epoch_training and epochs_completed != int(epochs):
        raise RuntimeError(
            f'fixed joint training stopped early: {epochs_completed}/{epochs}'
        )
    joint_protocol_dir = _experiment_path(args, f'fold_{fold}_JT')
    os.makedirs(joint_protocol_dir, exist_ok=True)
    joint_protocol = {
        'schema_version': 'kem-paper-strict5k-protocol-v1',
        'fold': int(fold), 'task': 'joint',
        'epochs_requested': int(epochs), 'epochs_completed': int(epochs_completed),
        'fixed_epoch_training': bool(args.fixed_epoch_training),
        'current_updates': int(joint_current_updates), 'replay_updates': 0,
        'stored_patches': 0, 'stored_wsi': 0, 'stored_pseudobags': 0,
        'gpu_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'gpu_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
        'cpu_peak_rss_kb': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        'cpu_peak_rss_delta_kb': max(
            0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            - joint_cpu_peak_start_kb,
        ),
        'epochs': joint_epoch_rows,
    }
    with open(os.path.join(
        joint_protocol_dir, 'protocol_audit.json'
    ), 'w', encoding='utf-8') as handle:
        json.dump(joint_protocol, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(joint_epoch_rows).to_csv(
        os.path.join(joint_protocol_dir, 'protocol_epochs.csv'), index=False,
    )

    del train_loader, val_loader, optimizer
    # testing, log acc and auc
    model.load_state_dict(torch.load(
        _experiment_log_path(args, f'fold_{fold}_JT.pt')
    ))
    model.eval()

    if args.n_classes == 2:
        test_acc_metric = fabric.to_device(Accuracy(task='binary'))
        test_acc_metric_task = fabric.to_device(Accuracy(task='binary'))
    else:
        test_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
        test_acc_metric_task = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
    
    test_dataloaders = fabric.setup_dataloaders(*datamodule['test_loader'])
    result = {'fold': fold}
    for idx, test_loader in enumerate(test_dataloaders):
        all_labels, all_predictions, all_probabilities = [], [], []
        for batch_idx, batch in enumerate(test_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break

            if args.net in ['clam_sb', 'clam_mb']:
                with torch.no_grad():
                    out = model(batch['features'])
                # bag_loss = F.cross_entropy(out['logits'], batch['label'])
                y_hat = out['logits'].argmax(dim=-1)
                y_prob = F.softmax(out['logits'], dim=-1)

                test_acc_metric.update(y_hat, batch['label'])
                test_acc_metric_task.update(y_hat, batch['label'])
                # test_auc_metric.update(y_prob, batch['label'])
            elif args.net in ['transmil', 'acmil', 'mambamil', 'chief_cached', 'wikg']:
                with torch.no_grad():
                    out = model(batch['features'])
                y_hat = out['logits'].argmax(dim=-1)
                y_prob = F.softmax(out['logits'], dim=-1)

                test_acc_metric.update(y_hat, batch['label'])
                test_acc_metric_task.update(y_hat, batch['label'])
                # test_auc_metric.update(y_prob, batch['label'])
            else:
                raise NotImplementedError
            all_labels.extend(batch['label'].detach().cpu().numpy().tolist())
            all_predictions.extend(y_hat.detach().cpu().numpy().tolist())
            all_probabilities.append(y_prob.detach().cpu().numpy())
        task_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
        macro_auroc, macro_f1 = compute_task_metrics(
            all_labels, all_predictions, np.concatenate(all_probabilities, axis=0), task_classes
        )
        result.update({
            f'{idx}_acc': test_acc_metric_task.compute().item(),
            f'{idx}_auroc': macro_auroc,
            f'{idx}_f1': macro_f1,
        })

        test_acc_metric_task.reset()
        # test_auc_metric.reset()

    result.update({'acc': test_acc_metric.compute().item()})

    print(result)
    df = pd.DataFrame(result, index=[0])
    df.to_csv(_experiment_log_path(args, f'fold_{fold}_results_JT.csv'), index=False)
    logger.finalize(f"Success on fold {fold}!")
    observability.close()
    return result

def test_on_one_fold(args, fold=0):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; CPU fallback is disabled')
    fabric = Fabric(devices=1, accelerator="cuda")
    model = load_model(args)
    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    results = []
    seen_classes = np.empty(0, dtype=int)
    for model_idx in range(args.n_tasks):
        cur_classes = np.asarray(np.unique(test_dataloaders[model_idx].dataset.targets), dtype=int)
        seen_classes = np.append(seen_classes, cur_classes).astype(int)
        unseen_classes = list(set(np.arange(args.n_classes)) - set(seen_classes))

        if args.load is not None:
            model.load_state_dict(torch.load(args.load, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(
                _experiment_log_path(args, f'fold_{fold}_task_{model_idx}.pt'),
                map_location='cpu',
            ))
        model = fabric.to_device(model)
        model.eval()

        result = {'fold': fold, 'task': model_idx}
        for idx, test_loader in enumerate(test_dataloaders):
            task_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
            all_labels, all_preds, all_probs = [], [], []
            for batch_idx, batch in enumerate(test_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break
                batch = fabric.to_device(batch)
                if args.net in ['clam_sb', 'clam_mb']:
                    with torch.no_grad():
                        out = model(batch['features'], seen_classes=seen_classes)
                elif args.net in ['transmil', 'mambamil', 'chief_cached']:
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100
                elif args.net == 'wikg':
                    with torch.no_grad():
                        out = wikg_continual_forward(model, batch['features'], seen_classes)
                else:
                    raise NotImplementedError

                y_hat = out['logits'].argmax(dim=-1)
                y_prob = F.softmax(out['logits'], dim=-1)
                all_labels.extend(batch['label'].detach().cpu().numpy().tolist())
                all_preds.extend(y_hat.detach().cpu().numpy().tolist())
                all_probs.append(y_prob.detach().cpu().numpy())

            accuracy = float(np.mean(np.asarray(all_preds) == np.asarray(all_labels)))
            if idx > model_idx:
                macro_auroc, macro_f1 = np.nan, 0.0
            else:
                macro_auroc, macro_f1 = compute_task_metrics(
                    all_labels,
                    all_preds,
                    np.concatenate(all_probs, axis=0),
                    task_classes,
                )
            result.update({
                f'{idx}_acc': accuracy,
                f'{idx}_auroc': macro_auroc,
                f'{idx}_f1': macro_f1,
            })
                
        print(result)
        results.append(result)

    return results

def test(args):
    seed_everything(args.seed)

    if args.load is None:
        results = []
        # Test models from all tasks
        for fold in range(args.folds_start, args.folds_end):
            result = test_on_one_fold(args, fold=fold)
            results.extend(result)
    else:
        fold = max(args.folds_start, 0)
        # model_fold = args.load.split('_')[-2]
        # assert fold == int(model_fold), f'Fold mismatch: {fold} vs {model_fold}'
        results = test_on_one_fold(args, fold=fold)

    # convert to csv
    df = pd.DataFrame(results)
    log_path = _experiment_log_path(args)
    log_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path / 'test_results.csv', index=False)

def main(args):
    log_path = _experiment_log_path(args)
    log_path.mkdir(parents=True, exist_ok=True)
    # save args to yaml
    with (log_path / 'args.yaml').open('w') as f:
        yaml.dump(vars(args), f)

    results = []
    for fold in range(args.folds_start, args.folds_end):
        if 'jt' in args.dataset or (hasattr(args, 'cl_method') and args.cl_method == 'joint'):
            result = one_fold_jt(args, fold=fold)
        else:
            result = one_fold(args, fold=fold)

        if isinstance(result, dict):
            results.append(result)
        else:
            results.extend(result)

    # convert to csv
    df = pd.DataFrame(results)
    df.to_csv(log_path / 'results.csv', index=False)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA GPU is required; CPU execution is disabled. '
            'Check CUDA_VISIBLE_DEVICES and the NVIDIA driver library path.'
        )
    args = init_args()

    if args.testing:
        test(args)
    else:
        main(args)
