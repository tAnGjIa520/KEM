"""Read-only MorphCL manifest and patient-disjoint splits."""
import csv
import hashlib
from pathlib import Path
import torch


class Corpus:
    def __init__(self, manifest, root, splits, fold):
        self.root = Path(root)
        with open(manifest) as stream:
            rows = list(csv.DictReader(stream))
        self.rows = {r['slide_id']: r for r in rows}
        if len(rows) != len(self.rows):
            raise ValueError('Duplicate slide IDs')
        self.hashes = {'manifest': hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}
        self.parts = []
        patients = {key: set() for key in ('train', 'val', 'test')}
        slides = {key: set() for key in patients}
        for task in range(5):
            path = Path(splits)/f'fold_{fold}'/f'splits_{task}.csv'
            self.hashes[f'task{task}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as stream:
                table = list(csv.DictReader(stream))
            parts = {}
            for part in patients:
                ids = [r[part] for r in table if r.get(part)]
                if not ids or len(set(ids)) != len(ids):
                    raise ValueError('Empty/duplicate partition')
                for sid in ids:
                    row = self.rows[sid]
                    if int(row['task_id']) != task or int(row['feature_dim']) != 1024:
                        raise ValueError('Task or UNI dimension mismatch')
                    patients[part].add(row['case_id'])
                    slides[part].add(sid)
                parts[part] = ids
            self.parts.append(parts)
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if patients[a] & patients[b] or slides[a] & slides[b]:
                raise ValueError('Patient/slide leakage')

    def label(self, sid):
        return int(self.rows[sid]['label_id'])

    def load(self, sid, cap=0):
        x = torch.load(self.root/self.rows[sid]['relative_path'], map_location='cpu', weights_only=True)
        if isinstance(x, dict):
            x = x['embedding']
        if x.ndim != 2 or x.shape[1] != 1024 or not len(x) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid UNI bag: {sid}')
        return (x[:cap] if cap else x).float()

    def subset(self, ids, per_class=0):
        if not per_class:
            return list(ids)
        counts, out = {}, []
        for sid in ids:
            label = self.label(sid)
            if counts.get(label, 0) < per_class:
                out.append(sid)
                counts[label] = counts.get(label, 0) + 1
        return out

    def conch_relative(self, sid):
        path = Path(self.rows[sid]['relative_path'])
        return Path(*['CONCH' if s == 'UNI' else 'h5_files' if s == 'pt_files' else s
                      for s in path.parts]).with_suffix('.h5')

    def uni_h5(self, sid):
        path = Path(self.rows[sid]['relative_path'])
        return self.root/Path(*['h5_files' if s == 'pt_files' else s for s in path.parts]).with_suffix('.h5')
