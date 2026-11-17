"""Generate frozen decoder word vectors from an explicit local PubMedBERT snapshot.

Run on the authorized GPU. No random-vector fallback and no implicit download.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


DIAGNOSES = [
    'normal tissue', 'tumor', 'esophageal adenocarcinoma',
    'esophageal squamous cell carcinoma', 'lung adenocarcinoma',
    'lung squamous cell carcinoma', 'colorectal adenocarcinoma',
    'stomach adenocarcinoma', 'chromophobe renal cell carcinoma',
    'clear cell renal cell carcinoma', 'papillary renal cell carcinoma',
]


def vocabulary():
    words = ['[CLS]', '[UNK]', '[SEP]', '[PAD]']
    sequences = {}
    for label, diagnosis in enumerate(DIAGNOSES):
        tokens = (diagnosis + '.').split()
        for token in tokens:
            if token not in words:
                words.append(token)
        sequences[label] = [0] + [words.index(t) for t in tokens] + [2]
    return words, sequences


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).cuda().eval()
    words, sequences = vocabulary()
    with torch.no_grad():
        encoded = tokenizer(words, padding=True, truncation=True, return_tensors='pt').to('cuda')
        vectors = model(**encoded).pooler_output.cpu()
    if vectors.shape != (len(words), 768) or not torch.isfinite(vectors).all():
        raise ValueError('Invalid PubMedBERT pooler vectors')
    hashes = {}
    for path in sorted(Path(args.model).glob('*')):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    h.update(block)
            hashes[path.name] = h.hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    torch.save(vectors, output / 'word_vectors.pt')
    (output / 'metadata.json').write_text(json.dumps({
        'source': 'neuml/pubmedbert-base-embeddings', 'revision': args.revision,
        'file_sha256': hashes, 'pooling': 'pooler_output', 'words': words,
        'class_sequences': sequences, 'diagnoses': DIAGNOSES,
        'note': 'MorphCL diagnosis vocabulary adaptation; not official author vectors.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
