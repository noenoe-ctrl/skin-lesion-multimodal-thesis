"""Read-only reconciliation of all full-run predictions and recorded metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from protocol import CLASSES, DEFAULT_OUTPUT, SEEDS, digest, write_json
from run_experiment import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_OUTPUT / 'runs' / 'paper_baseline_v1')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and manifest['completed_experiments'] == 90
    assert manifest['seeds'] == SEEDS and not manifest['factorial']
    metrics = pd.read_csv(run / 'metrics_per_seed.csv')
    expected = {(s, t, m) for s in SEEDS for t in ['binary', 'multi'] for m in MODELS}
    assert len(metrics) == 90 and set(zip(metrics.seed, metrics.task, metrics.model)) == expected
    assert not metrics[['seed', 'task', 'model']].duplicated().any()
    assert (metrics.evaluated_split == 'test').all() and not metrics.smoke.any()
    sources = {}
    for _, row in metrics.iterrows():
        folder = run / f'seed_{row.seed}' / row.task / f'{row.model}_{row.loss}'
        p = pd.read_csv(folder / 'test_predictions.csv')
        test = pd.read_csv(run.parents[1] / 'splits' / f'seed_{row.seed}' / 'test_raw.csv')
        assert p.img_id.tolist() == test.img_id.tolist()
        assert p.patient_id.tolist() == test.patient_id.tolist()
        assert p.lesion_id.tolist() == test.lesion_id.tolist()
        assert p.diagnostic.tolist() == test.diagnostic.tolist()
        binary = row.task == 'binary'
        truth = test.label.to_numpy() if binary else test.diagnostic.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
        np.testing.assert_array_equal(p.target, truth)
        prob = p.prob_malignant.to_numpy() if binary else p[[f'prob_{c}' for c in CLASSES]].to_numpy()
        assert np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1))
        if not binary:
            np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-6)
        pred = (prob >= .5).astype(int) if binary else prob.argmax(axis=1)
        np.testing.assert_array_equal(p.prediction, pred)
        avg = 'binary' if binary else 'macro'
        recalculated = {
            'auc': roc_auc_score(truth, prob) if binary else roc_auc_score(truth, prob, multi_class='ovr', average='macro'),
            'accuracy': accuracy_score(truth, pred),
            'balanced_accuracy': balanced_accuracy_score(truth, pred),
            'precision': precision_score(truth, pred, average=avg, zero_division=0),
            'recall': recall_score(truth, pred, average=avg, zero_division=0),
            'f1': f1_score(truth, pred, average=avg, zero_division=0)}
        for metric, value in recalculated.items():
            np.testing.assert_allclose(row[metric], value, rtol=0, atol=1e-8)
        history = pd.read_csv(folder / 'history.csv')
        assert row.best_epoch == history.loc[history.selection_score.idxmax(), 'epoch']
        assert history[['train_loss','val_loss','selection_score']].apply(np.isfinite).all().all()
        assert (folder / 'head.pt').stat().st_size > 0
        sources[str((folder / 'test_predictions.csv').relative_to(run))] = digest(folder / 'test_predictions.csv')
    summary = pd.read_csv(run / 'summary.csv').set_index(['task', 'model', 'loss'])
    for key, group in metrics.groupby(['task', 'model', 'loss']):
        assert len(group) == 5
        for metric in ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
            np.testing.assert_allclose(summary.loc[key, metric + '_mean'], group[metric].mean(), atol=1e-12)
            np.testing.assert_allclose(summary.loc[key, metric + '_std'], group[metric].std(ddof=0), atol=1e-12)
    report = {'status':'passed', 'experiments':90, 'summary_groups':18,
              'metrics_recomputed_from_test_predictions':True,
              'predictions_match_test_manifests':True, 'best_epochs_match_validation_history':True,
              'all_values_finite':True, 'aggregation_ddof':0,
              'metrics_sha256':digest(run / 'metrics_per_seed.csv'),
              'summary_sha256':digest(run / 'summary.csv'), 'prediction_sha256':sources}
    write_json(run / 'results_verification.json', report)
    print('VERIFIED: 90 runs; six metrics recomputed; manifests and 18 aggregates agree.')


if __name__ == '__main__':
    main()
