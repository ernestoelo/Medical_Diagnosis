#!/usr/bin/env python3
"""Run a small ItôE inference demo using the converted `itoe_model.pth`.

Usage:
    python tools/run_itoe_demo.py --model-dir Medical_Diagnosis_Assistant/itoe_model

This script:
 - Loads `itoe_metadata.json` to map entity names to ids
 - Loads the ItôE model via `src.itoe_inference.load_itoe_model`
 - Finds a symptom id by searching for 'fever' or 'fiebre' in the vocabulary
 - Scores all candidate target entities and prints top-K ranked results
"""
import os
import argparse
import json
import torch
import sys

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.itoe_inference import load_itoe_model


def find_symptom_ids(ent2id, queries=('fever', 'fiebre')):
    qlower = [q.lower() for q in queries]
    matches = {}
    for ent, idx in ent2id.items():
        el = ent.lower()
        for q in qlower:
            if q in el:
                matches[ent] = idx
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='Medical_Diagnosis_Assistant/itoe_model')
    parser.add_argument('--metadata', default='Medical_Diagnosis_Assistant/itoe_metadata.json')
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    metadata_path = os.path.abspath(args.metadata)

    if not os.path.exists(metadata_path):
        raise SystemExit(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    ent2id = metadata.get('ent2id', {})
    rel2id = metadata.get('rel2id', {})
    dim = metadata.get('dim')

    print(f"Vocabulary size: {len(ent2id)}  rels: {rel2id}  dim: {dim}")

    # Find symptom candidates
    matches = find_symptom_ids(ent2id, queries=('fever', 'fiebre'))
    if not matches:
        # fallback: try common label
        fallback = 'Fever  unspecified'
        if fallback in ent2id:
            matches = {fallback: ent2id[fallback]}

    if not matches:
        raise SystemExit('No symptom entity matching "fever"/"fiebre" found in metadata.')

    print('Found symptom matches (sample):')
    for ent, idx in list(matches.items())[:5]:
        print(' -', ent, '->', idx)

    # Load model
    print('Loading ItôE model...')
    model = load_itoe_model(model_dir, metadata_path=metadata_path, map_location='cpu')

    model.eval()

    # Use the first matched symptom for demo
    symptom_ent, symptom_idx = next(iter(matches.items()))
    h_idx = torch.tensor([symptom_idx], dtype=torch.long)

    # Choose relation 'association' if present, else first rel
    rel_name = 'association'
    if rel_name in rel2id:
        r_idx = torch.tensor([rel2id[rel_name]], dtype=torch.long)
    else:
        r_idx = torch.tensor([list(rel2id.values())[0]], dtype=torch.long)

    num_ent = len(ent2id)
    t_idx = torch.arange(num_ent, dtype=torch.long)

    # Expand h and r to match t batch
    h_batch = h_idx.repeat(num_ent)
    r_batch = r_idx.repeat(num_ent)

    with torch.no_grad():
        energies = model.calcular_energia(h_batch, r_batch, t_idx)

    # Lower energy -> more plausible; convert to scores via softmax over -energy
    scores = torch.softmax(-energies, dim=0)

    topk = args.topk
    vals, idxs = torch.topk(scores, topk)

    # Build inverse mapping
    id2ent = {v: k for k, v in ent2id.items()}

    print('\nTop candidates for symptom:', symptom_ent)
    for rank, (s, i) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        ent_name = id2ent.get(i, f'<id:{i}>')
        print(f'{rank:2d}. {ent_name} (id={i}) — score={s:.4f}')


if __name__ == '__main__':
    main()
