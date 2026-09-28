"""Explicit CLI: prepare, verify, smoke or full. No full training by default."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import (CLASSES, DEFAULT_DATA, DEFAULT_OUTPUT, TrainMetadata,
                      digest, embedding_index, prepare, verify, write_json)

MODELS = {
    'clip': ('clip',), 'dino': ('dino',), 'convnext': ('convnext',),
    'clip_dino': ('clip', 'dino'), 'clip_meta': ('clip', 'meta'),
    'dino_meta': ('dino', 'meta'), 'convnext_meta': ('convnext', 'meta'),
    'clip_dino_meta': ('clip', 'dino', 'meta'),
    'fusion': ('clip', 'dino', 'meta'),
}


def small_sample(df, cap, seed):
    """Diagnostic class-balanced sample for smoke only; not paper evaluation."""
    rng = np.random.default_rng(seed)
    by_class = [rng.permutation(np.flatnonzero(df.diagnostic.to_numpy() == c)).tolist() for c in CLASSES]
    chosen = []
    while len(chosen) < min(cap, len(df)):
        for indices in by_class:
            if indices and len(chosen) < cap:
                chosen.append(indices.pop())
    return df.iloc[chosen].reset_index(drop=True)


def train_one(parts, metadata, visual, image_lookup, task, name, loss_name,
              seed, epochs, patience, device_name, dest, smoke=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    binary = task == 'binary'
    output_dim = 1 if binary else len(CLASSES)
    ymap = {c: i for i, c in enumerate(CLASSES)}
    dimensions = {'clip': 768, 'dino': 1024, 'convnext': 1024,
                  'meta': len(metadata.state['feature_names'])}
    modalities = MODELS[name]
    fusion = name == 'fusion'

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(sum(dimensions[m] for m in modalities), 512),
                                     nn.ReLU(), nn.Dropout(.3), nn.Linear(512, 256),
                                     nn.ReLU(), nn.Dropout(.3), nn.Linear(256, output_dim))
        def forward(self, *x):
            return self.net(torch.cat(x, dim=1))

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(.3))
            self.dino = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(.3))
            self.meta = nn.Sequential(nn.Linear(dimensions['meta'], 128), nn.ReLU(), nn.Dropout(.3),
                                      nn.Linear(128, 256), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(.3),
                                      nn.Linear(256, output_dim))
        def forward(self, c, d, m):
            return self.head(torch.cat([self.dino(d), self.clip(c), self.meta(m)], dim=1))

    def labels(p):
        return p.label.to_numpy(dtype=np.float32) if binary else p.diagnostic.map(ymap).to_numpy(dtype=np.int64)

    def loader(p, shuffle=False):
        indices = [image_lookup[img] for img in p.img_id]
        features = [metadata.transform(p) if m == 'meta' else visual[m][indices] for m in modalities]
        tensors = [torch.tensor(x, dtype=torch.float32) for x in features]
        tensors.append(torch.tensor(labels(p), dtype=torch.float32 if binary else torch.long))
        return DataLoader(TensorDataset(*tensors), batch_size=64, shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))

    train_loader = loader(parts['train'], True)
    val_loader = loader(parts['val'])
    weights = None
    if not binary:
        counts = np.bincount(labels(parts['train']), minlength=6)
        if np.any(counts == 0):
            raise ValueError('Training subset is missing a class')
        weights = torch.tensor(len(parts['train']) / (6 * counts), dtype=torch.float32, device=device)

    def criterion(logits, y):
        if binary:
            logits = logits.squeeze(1)
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            if loss_name == 'bce':
                return bce.mean()
            prob = torch.sigmoid(logits)
            pt = prob * y + (1 - prob) * (1 - y)
            alpha = .25 * y + .75 * (1 - y)
            return (alpha * (1 - pt).pow(2) * bce).mean()
        # Correct standard focal loss: weighting stays outside probability.
        logpt = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        return (-weights[y] * (1 - logpt.exp()).pow(2) * logpt).mean()

    def evaluate(model, batches):
        model.eval(); probs = []; truths = []; losses = []; sizes = []
        with torch.no_grad():
            for batch in batches:
                *x, y = [t.to(device) for t in batch]
                logits = model(*x)
                losses.append(float(criterion(logits, y))); sizes.append(len(y))
                p = torch.sigmoid(logits.squeeze(1)) if binary else torch.softmax(logits, dim=1)
                probs.append(p.cpu().numpy()); truths.append(y.cpu().numpy())
        p, y = np.concatenate(probs), np.concatenate(truths)
        assert np.isfinite(p).all()
        pred = (p >= .5).astype(int) if binary else p.argmax(axis=1)
        score = roc_auc_score(y, p) if binary else balanced_accuracy_score(y, pred)
        return score, float(np.average(losses, weights=sizes)), p, pred, y

    model = (Fusion() if fusion else Head()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    best = -np.inf; state = None; waiting = 0; history = []; best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.; count = 0
        for batch in train_loader:
            *x, y = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(*x), y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(y); count += len(y)
        score, vloss, _, _, _ = evaluate(model, val_loader)
        scheduler.step(vloss)
        history.append({'epoch': epoch, 'train_loss': total / count, 'val_loss': vloss,
                        'selection_score': score})
        if score > best:
            best = score; waiting = 0; best_epoch = epoch
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(state)
    # Smoke never creates a test loader or generates test predictions.
    evaluated_split = 'val' if smoke else 'test'
    _, _, prob, pred, true = evaluate(model, val_loader if smoke else loader(parts['test']))
    avg = 'binary' if binary else 'macro'
    metrics = {'auc': float(roc_auc_score(true, prob) if binary else
                             roc_auc_score(true, prob, multi_class='ovr', average='macro')),
               'accuracy': float(accuracy_score(true, pred)),
               'balanced_accuracy': float(balanced_accuracy_score(true, pred)),
               'precision': float(precision_score(true, pred, average=avg, zero_division=0)),
               'recall': float(recall_score(true, pred, average=avg, zero_division=0)),
               'f1': float(f1_score(true, pred, average=avg, zero_division=0)),
               'seed': seed, 'task': task, 'model': name, 'loss': loss_name,
               'evaluated_split': evaluated_split, 'smoke': smoke, 'best_epoch': best_epoch,
               'best_validation_score': float(best), 'train_rows': len(parts['train']),
               'val_rows': len(parts['val']), 'evaluated_rows': len(true)}
    dest.mkdir(parents=True, exist_ok=False)
    torch.save(state, dest / 'head.pt')
    write_json(dest / 'metrics.json', metrics)
    write_json(dest / 'configuration.json', {'seed': seed, 'task': task, 'model': name,
               'loss': loss_name, 'epochs_limit': epochs, 'patience': patience, 'batch_size': 64,
               'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': .3, 'device': str(device),
               'torch': torch.__version__, 'deterministic_algorithms': True,
               'train_img_ids': parts['train'].img_id.tolist(),
               'val_img_ids': parts['val'].img_id.tolist(),
               'preprocessor': metadata.state, 'parameter_count': sum(p.numel() for p in model.parameters())})
    pd.DataFrame(history).to_csv(dest / 'history.csv', index=False)
    predicted = parts[evaluated_split][['img_id', 'patient_id', 'lesion_id', 'diagnostic']].copy()
    predicted['target'] = true; predicted['prediction'] = pred
    if binary:
        predicted['prob_malignant'] = prob
    else:
        for i, c in enumerate(CLASSES):
            predicted[f'prob_{c}'] = prob[:, i]
    predicted.to_csv(dest / f'{evaluated_split}_predictions.csv', index=False)
    print(f'{task}/{name}/{loss_name}: {len(history)} epochs; {evaluated_split}; finite outputs')
    return metrics


def run(args):
    # Must be set before importing torch / initializing CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(2)
    df, config = verify(args.data_dir, args.output)
    visual, _ = embedding_index(args.data_dir, df)
    lookup = {img: i for i, img in enumerate(df.img_id)}
    smoke = args.command == 'smoke'
    run_dir = args.output / ('smoke' if smoke else 'runs') / args.run_id
    if run_dir.exists():
        raise FileExistsError(f'Run already exists: {run_dir}; choose a new --run-id')
    chosen_models = args.models or (['clip_dino_meta', 'fusion'] if smoke else list(MODELS))
    if set(chosen_models) - set(MODELS):
        raise ValueError('Unknown model selection')
    seeds = [config['seeds'][0]] if smoke else config['seeds']
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'run_manifest.json', {'command': args.command,
               'status': 'running', 'seeds': seeds, 'models': chosen_models,
               'tasks': args.tasks, 'factorial': args.factorial,
               'protocol_sha256': digest(args.output / 'protocol.json'),
               'code_sha256': {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('protocol.py')]},
               'argv': sys.argv, 'test_evaluated': not smoke})
    results = []
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in seeds:
        folder = args.output / 'splits' / f'seed_{seed}'
        # Smoke deliberately never loads test_raw.csv into its training flow.
        splits = ['train', 'val'] if smoke else ['train', 'val', 'test']
        parts = {s: pd.read_csv(folder / f'{s}_raw.csv') for s in splits}
        processor = TrainMetadata(json.loads((folder / 'preprocessor.json').read_text(encoding='utf-8')))
        if smoke:
            parts = {s: small_sample(p, 128 if s == 'train' else 64, seed) for s, p in parts.items()}
        for task in args.tasks:
            experiments = [(name, 'focal' if task == 'multi' or name == 'fusion' else 'bce')
                           for name in chosen_models]
            if args.factorial and task == 'binary':
                experiments += [('clip_dino_meta', 'focal'), ('fusion', 'bce')]
            for name, loss_name in experiments:
                metrics = train_one(parts, processor, visual, lookup, task, name, loss_name,
                                    seed, 2 if smoke else args.epochs, 2 if smoke else 12,
                                    device, run_dir / f'seed_{seed}' / task / f'{name}_{loss_name}', smoke)
                results.append(metrics)
                pd.DataFrame(results).to_csv(run_dir / 'metrics_per_seed.csv', index=False)
    metrics = pd.DataFrame(results)
    cols = ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']
    if not smoke:
        agg = metrics.groupby(['task', 'model', 'loss'])[cols].agg(['mean', lambda x: x.std(ddof=0)])
        agg.columns = [f'{a}_{"mean" if b == "mean" else "std"}' for a, b in agg.columns]
        agg.to_csv(run_dir / 'summary.csv')
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='complete', completed_experiments=len(results))
    write_json(run_dir / 'run_manifest.json', manifest)
    print(f'Saved {len(results)} experiments in {run_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'smoke', 'full'])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run-id', default='baseline_v1')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--models', nargs='+', choices=list(MODELS))
    parser.add_argument('--tasks', nargs='+', choices=['binary', 'multi'], default=['binary', 'multi'])
    parser.add_argument('--factorial', action='store_true', help='Add concat+focal and fusion+BCE in binary')
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve(); args.output = args.output.resolve()
    if args.output == args.data_dir or args.data_dir in args.output.parents:
        parser.error('Output must be outside historical dataset directory')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('--run-id must contain only letters, numbers, underscore or hyphen')
    if len(set(args.tasks)) != len(args.tasks) or (args.models and len(set(args.models)) != len(args.models)):
        parser.error('Duplicate task/model selection')
    if args.command == 'prepare':
        print(prepare(args.data_dir, args.output).to_string(index=False))
    elif args.command == 'verify':
        verify(args.data_dir, args.output)
    else:
        run(args)


if __name__ == '__main__':
    main()
