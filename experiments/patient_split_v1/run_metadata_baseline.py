"""Metadata-only extension; reuses the existing splits and training procedure.

No original runner, notebook, split, preprocessor or result is overwritten.
Only an explicit `full` command evaluates test; `smoke` evaluates validation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

import run_experiment as shared
from protocol import (DEFAULT_DATA, DEFAULT_OUTPUT, META, TrainMetadata,
                      digest, verify, write_json)

MODEL = 'metadata_mlp'


def train_metadata(parts, processor, task, loss, seed, epochs, patience,
                   device, dest, smoke=False):
    # Register only in memory. The original nine-model CLI remains unchanged.
    shared.MODELS[MODEL] = ('meta',)
    lookup = {img: i for i, img in enumerate(parts['train'].img_id)}
    for part in parts.values():
        lookup.update({img: i for i, img in enumerate(part.img_id)})
    # An empty visual dictionary deliberately makes any visual access fail.
    return shared.train_one(parts, processor, {}, lookup, task, MODEL, loss,
                            seed, epochs, patience, device, dest, smoke)


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(2)
    # Audits the existing data/splits/cache hashes, but does not train a visual model.
    _, config = verify(args.data_dir, args.output)
    smoke = args.command == 'smoke'
    seeds = config['seeds'][:1] if smoke else config['seeds']
    destination = args.output / ('smoke' if smoke else 'runs') / args.run_id
    if destination.exists():
        raise FileExistsError(f'Run already exists: {destination}; choose another --run-id')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    destination.mkdir(parents=True)
    sources = [Path(__file__), Path(shared.__file__), Path(__file__).with_name('protocol.py')]
    manifest = {'status': 'running', 'command': args.command, 'seeds': seeds,
                'models': [MODEL], 'tasks': args.tasks, 'binary_loss': args.binary_loss,
                'input_modalities': ['meta'], 'clinical_columns': META,
                'test_evaluated': not smoke, 'device': device, 'argv': sys.argv,
                'protocol_sha256': digest(args.output / 'protocol.json'),
                'code_sha256': {p.name: digest(p) for p in sources},
                'partition_sha256': {},
                'comparison_reference': 'paper_baseline_v1',
                'selection': {'binary': 'validation ROC-AUC', 'multi': 'validation BACC'},
                'smoke_note': 'Not a scientific result; train subset uses full-train fitted preprocessing.' if smoke else None}
    write_json(destination / 'run_manifest.json', manifest)
    results = []
    for seed in seeds:
        folder = args.output / 'splits' / f'seed_{seed}'
        names = ['train', 'val'] if smoke else ['train', 'val', 'test']
        parts = {s: pd.read_csv(folder / f'{s}_raw.csv') for s in names}
        processor = TrainMetadata(json.loads((folder / 'preprocessor.json').read_text(encoding='utf-8')))
        manifest['partition_sha256'][str(seed)] = {
            p.name: digest(p) for p in [folder / 'manifest.csv', folder / 'preprocessor.json']
        }
        if smoke:
            parts = {s: shared.small_sample(p, 128 if s == 'train' else 64, seed)
                     for s, p in parts.items()}
        for task in args.tasks:
            loss = args.binary_loss if task == 'binary' else 'focal'
            result = train_metadata(parts, processor, task, loss, seed,
                                    2 if smoke else args.epochs, 2 if smoke else 12,
                                    device, destination / f'seed_{seed}' / task / f'{MODEL}_{loss}', smoke)
            results.append(result)
            pd.DataFrame(results).to_csv(destination / 'metrics_per_seed.csv', index=False)
    if not smoke:
        columns = ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']
        agg = pd.DataFrame(results).groupby(['task', 'model', 'loss'])[columns].agg(
            ['mean', lambda x: x.std(ddof=0)])
        agg.columns = [f'{a}_{"mean" if b == "mean" else "std"}' for a, b in agg.columns]
        agg.to_csv(destination / 'summary.csv')
    manifest.update(status='complete', completed_experiments=len(results))
    write_json(destination / 'run_manifest.json', manifest)
    print(f'Saved {len(results)} metadata-only experiments in {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'full'])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run-id', default='metadata_only_v1')
    parser.add_argument('--tasks', nargs='+', choices=['binary', 'multi'], default=['binary', 'multi'])
    parser.add_argument('--binary-loss', choices=['bce', 'focal'], default='bce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve(); args.output = args.output.resolve()
    if args.output == args.data_dir or args.data_dir in args.output.parents:
        parser.error('Output must be outside historical dataset directory')
    if args.epochs < 1 or len(set(args.tasks)) != len(args.tasks):
        parser.error('Positive epochs and unique tasks are required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Use only letters, numbers, underscore or hyphen in run-id')
    run(args)


if __name__ == '__main__':
    main()
