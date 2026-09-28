"""Reconcile metadata-only outputs against the unchanged partition artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from protocol import CLASSES, DEFAULT_OUTPUT, digest, write_json


def check(run, output):
    manifest = json.loads((run / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['input_modalities'] == ['meta']
    assert manifest['protocol_sha256'] == digest(output / 'protocol.json')
    smoke = manifest['command'] == 'smoke'
    split = 'val' if smoke else 'test'
    rows = pd.read_csv(run / 'metrics_per_seed.csv')
    expected = {(s, t) for s in manifest['seeds'] for t in manifest['tasks']}
    assert len(rows) == len(expected) == manifest['completed_experiments']
    assert set(zip(rows.seed, rows.task)) == expected
    assert (rows.model == 'metadata_mlp').all()
    reference = output / 'runs' / 'paper_baseline_v1'
    for _, row in rows.iterrows():
        folder = output / 'splits' / f'seed_{row.seed}'
        for filename, value in manifest['partition_sha256'][str(row.seed)].items():
            assert digest(folder / filename) == value
        dest = run / f'seed_{row.seed}' / row.task / f'metadata_mlp_{row.loss}'
        saved = json.loads((dest / 'configuration.json').read_text())
        state = json.loads((folder / 'preprocessor.json').read_text())
        assert saved['preprocessor'] == state
        # Same fitted state as the previously trained A model, if available.
        ref_loss = 'bce' if row.task == 'binary' else 'focal'
        ref_config = reference / f'seed_{row.seed}' / row.task / f'clip_dino_meta_{ref_loss}' / 'configuration.json'
        if ref_config.exists():
            assert json.loads(ref_config.read_text())['preprocessor'] == state
        for name in ['train', 'val']:
            ids = pd.read_csv(folder / f'{name}_raw.csv').img_id.tolist()
            selected = saved[f'{name}_img_ids']
            assert set(selected) <= set(ids) if smoke else selected == ids
        prediction = pd.read_csv(dest / f'{split}_predictions.csv')
        raw = pd.read_csv(folder / f'{split}_raw.csv').set_index('img_id')
        assert not prediction.img_id.duplicated().any()
        if not smoke:
            assert prediction.img_id.tolist() == raw.index.tolist()
        actual = raw.loc[prediction.img_id]
        for name in ['patient_id', 'lesion_id', 'diagnostic']:
            assert prediction[name].tolist() == actual[name].tolist()
        binary = row.task == 'binary'
        assert row.loss == (manifest['binary_loss'] if binary else 'focal')
        truth = actual.label.to_numpy() if binary else actual.diagnostic.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
        np.testing.assert_array_equal(prediction.target, truth)
        probs = prediction.prob_malignant.to_numpy() if binary else prediction[[f'prob_{c}' for c in CLASSES]].to_numpy()
        assert np.isfinite(probs).all() and np.all((probs >= 0) & (probs <= 1))
        if not binary:
            np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
        pred = (probs >= .5).astype(int) if binary else probs.argmax(axis=1)
        np.testing.assert_array_equal(prediction.prediction, pred)
        avg = 'binary' if binary else 'macro'
        metrics = {'auc': roc_auc_score(truth, probs) if binary else roc_auc_score(truth, probs, multi_class='ovr', average='macro'),
                   'accuracy': accuracy_score(truth, pred), 'balanced_accuracy': balanced_accuracy_score(truth, pred),
                   'precision': precision_score(truth, pred, average=avg, zero_division=0),
                   'recall': recall_score(truth, pred, average=avg, zero_division=0),
                   'f1': f1_score(truth, pred, average=avg, zero_division=0)}
        for key, value in metrics.items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-8)
        history = pd.read_csv(dest / 'history.csv')
        assert row.best_epoch == history.loc[history.selection_score.idxmax(), 'epoch']
        assert history[['train_loss', 'val_loss', 'selection_score']].apply(np.isfinite).all().all()
        if smoke:
            np.testing.assert_allclose(row.best_validation_score, metrics['auc' if binary else 'balanced_accuracy'], atol=1e-8)
            assert not list(dest.glob('test*'))
        assert (dest / 'head.pt').stat().st_size > 0
    if not smoke:
        summary = pd.read_csv(run / 'summary.csv').set_index(['task', 'model', 'loss'])
        for key, group in rows.groupby(['task', 'model', 'loss']):
            for metric in ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
                np.testing.assert_allclose(summary.loc[key, metric + '_mean'], group[metric].mean(), atol=1e-12)
                np.testing.assert_allclose(summary.loc[key, metric + '_std'], group[metric].std(ddof=0), atol=1e-12)
    report = {'status': 'passed', 'experiments': len(rows), 'smoke': smoke,
              'test_evaluated': not smoke, 'metrics_recomputed': True,
              'partition_hashes_match': True, 'preprocessor_matches_existing_A': True,
              'best_epochs_match_validation': True, 'metadata_only': True}
    write_json(run / 'results_verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    check(args.run_dir.resolve(), args.output.resolve())
