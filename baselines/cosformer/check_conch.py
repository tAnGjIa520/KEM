"""Check actual pretrained CONCH projection/text interface without training.

This verifies mathematical projection equivalence and feature/coordinate shapes;
it does NOT by itself certify the historical encoder provenance of a cached H5.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom.custom_tokenizer import get_tokenizer, tokenize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--conch-h5', required=True)
    p.add_argument('--uni-h5', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    destination = Path(a.output)
    if destination.exists():
        raise FileExistsError(destination)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    model, _ = create_model_from_pretrained('conch_ViT-B-16', a.checkpoint, device='cuda')
    model.eval()
    tokenizer = get_tokenizer()
    names = ['normal tissue', 'tumor', 'esophageal adenocarcinoma',
             'esophageal squamous cell carcinoma', 'lung adenocarcinoma',
             'lung squamous cell carcinoma', 'colorectal adenocarcinoma',
             'stomach adenocarcinoma', 'chromophobe renal cell carcinoma',
             'clear cell renal cell carcinoma', 'papillary renal cell carcinoma']
    prompts = [f'a histopathology image of {name}.' for name in names]
    with h5py.File(a.conch_h5, 'r') as c, h5py.File(a.uni_h5, 'r') as u:
        x = torch.from_numpy(c['features'][:]).float().cuda()
        same_coords = np.array_equal(c['coords'][:], u['coords'][:])
        shapes = {'conch': list(x.shape), 'uni': list(u['features'].shape)}
    if not same_coords:
        raise ValueError('Cached CONCH/UNI patch coordinates differ')
    with torch.no_grad():
        text = model.encode_text(tokenize(texts=prompts, tokenizer=tokenizer).cuda())
        projected = torch.nn.functional.normalize(model.visual.forward_project(x), dim=-1)
        if projected.shape[1] != text.shape[1]:
            raise ValueError('Contrastive dimensions differ')
        scores = projected @ text.T
        torch.manual_seed(1)
        dummy = torch.randn(1, 3, 448, 448, device='cuda')
        raw = model.encode_image(dummy, proj_contrast=False, normalize=False)
        direct = model.encode_image(dummy, proj_contrast=True, normalize=True)
        two_stage = torch.nn.functional.normalize(model.visual.forward_project(raw), dim=-1)
        error = float((direct - two_stage).abs().max())
        if error > 1e-5 or not torch.isfinite(scores).all():
            raise ValueError('Projection parity or finite scores failed')
    h = hashlib.sha256()
    with open(a.checkpoint, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    result = {'checkpoint_sha256': h.hexdigest(), 'same_coords': same_coords,
              'shapes': shapes, 'projected_dimension': projected.shape[1],
              'projection_parity_max_error': error, 'prompts': prompts,
              'score_min': scores.min().item(), 'score_max': scores.max().item(),
              'historical_feature_provenance_verified': False,
              'note': 'Interface check only; no trained COSFormer result.'}
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
