"""CONCH-aligned, train-only COSFormer retrieval adaptation.

Inputs are contrastively projected image and text features from the SAME CONCH
checkpoint, not UNI vectors or the pre-projection CONCH visual representation.
This module does not claim to reproduce missing upstream alignment code.
"""
import numpy as np
from sklearn.cluster import KMeans


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite feature matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError('Zero feature vector')
    return x / norm


def select_representatives(bags, class_text, allowed_train_ids, clusters_per_class=2, seed=1):
    """bags: iterable (slide_id, global_label, projected_CONCH_patches).

    Explicit adaptation: cluster normalized mean patch embeddings by class;
    rank each cluster by maximum patch-to-class-text cosine similarity.
    Class text is the normalized average of predeclared template embeddings.
    """
    if clusters_per_class < 1:
        raise ValueError('Invalid cluster count')
    allowed, visited, groups = set(allowed_train_ids), set(), {}
    for sid, label, patches in bags:
        if sid not in allowed or sid in visited:
            raise ValueError('Non-training or duplicate slide in selector')
        visited.add(sid)
        x = normalized(patches)
        text = normalized(np.asarray(class_text[int(label)]).reshape(1, -1))[0]
        if x.shape[1] != len(text):
            raise ValueError('Image/text dimensions differ; arbitrary projection is forbidden')
        center = normalized(x.mean(0, keepdims=True))[0]
        groups.setdefault(int(label), []).append((sid, center, float(np.max(x @ text))))
    if not groups:
        raise ValueError('Empty training selector input')
    selected, audit = [], []
    for label, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row[0])
        values = np.stack([row[1] for row in rows])
        k = min(clusters_per_class, len(rows), len(np.unique(values, axis=0)))
        assignment = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(values)
        for cluster in sorted(set(assignment)):
            candidates = [rows[i] for i in np.flatnonzero(assignment == cluster)]
            chosen = sorted(candidates, key=lambda row: (-row[2], row[0]))[0]
            selected.append(chosen[0])
            audit.append({'slide_id': chosen[0], 'label': label, 'cluster': int(cluster),
                          'score': chosen[2], 'cluster_size': len(candidates),
                          'requested_clusters': clusters_per_class, 'actual_clusters': int(k)})
    return selected, audit


def merge_surviving_buffer(previous, new, capacity, rng):
    """Only surviving exemplars and newly selected current-task records enter.

    Each record includes slide_id and retained tensors; callers must not replace
    tensors with unrestricted historical-file references. Overflow is random,
    as in the published buffer, not a KEM coverage policy.
    """
    if capacity < 1:
        raise ValueError('Invalid memory capacity')
    merged = list(previous) + list(new)
    ids = [item['slide_id'] for item in merged]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate retained/new slide')
    if len(merged) > capacity:
        indices = sorted(rng.choice(len(merged), capacity, replace=False))
        merged = [merged[i] for i in indices]
    return merged
