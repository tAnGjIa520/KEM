import csv
import tempfile
import unittest
from pathlib import Path

import torch

from train_morphcl import ABMIL, Corpus


class TrainingTests(unittest.TestCase):
    def test_model_mask_and_gradients(self):
        torch.manual_seed(1)
        model = ABMIL()
        logits, attention = model(torch.randn(13, 1024), {0, 1})
        self.assertTrue(torch.isneginf(logits[2:]).all())
        self.assertAlmostEqual(attention.sum().item(), 1.0, places=5)
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.project[0].weight.grad).all())

    def test_patient_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'fold_0').mkdir()
            fields = ['slide_id', 'case_id', 'task_id', 'feature_dim', 'label_id', 'relative_path']
            with (root / 'manifest.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for task in range(5):
                    for part in ['train', 'val', 'test']:
                        sid = f'{task}_{part}'
                        writer.writerow(dict(slide_id=sid, case_id=f'{task}_same_patient',
                            task_id=task, feature_dim=1024, label_id=task, relative_path=sid+'.pt'))
                    (root / 'fold_0' / f'splits_{task}.csv').write_text(
                        f'train,val,test\n{task}_train,{task}_val,{task}_test\n')
            with self.assertRaisesRegex(ValueError, 'Patient/slide leakage'):
                Corpus(root / 'manifest.csv', root, root, 0)


if __name__ == '__main__':
    unittest.main()
