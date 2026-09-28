"""Additional clinical references and loss/capacity controls; no old runs overwritten."""
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

import run_experiment as shared
from protocol import (CLASSES, DEFAULT_DATA, DEFAULT_OUTPUT, SEEDS, TrainMetadata,
                      digest, embedding_index, verify, write_json)
from run_metadata_baseline import train_metadata
from thesis_metrics import evaluate_probabilities, target, prediction_frame

DEST = DEFAULT_OUTPUT / 'runs' / 'thesis_completion_v1'
PLAN = {'clinical_models': ['metadata_logreg', 'metadata_boosting', 'metadata_prior'],
        'logistic_C': [0.01, 0.1, 1., 10.],
        'boosting_grid': [{'max_iter': n, 'max_leaf_nodes': leaves} for n in [100, 200] for leaves in [7, 15]],
        'boosting_fixed': {'learning_rate': .1, 'min_samples_leaf': 20, 'l2_regularization': 1., 'early_stopping': False},
        'selection': {'binary': 'validation AUC', 'multi': 'validation BACC'},
        'refit_train_plus_val': False,
        'clinical_selector': ['metadata_mlp', 'metadata_logreg', 'metadata_boosting'],
        'selector_tie_order': 'clinical_selector order',
        'controls': 'binary: A focal, B BCE, clinical MLP focal; B same-parameter metadata-only and visual-only with focal in both tasks',
        'bootstrap': {'draws': 2000, 'seed': 20260928, 'cluster': 'patient_id',
                      'within_split': 'multinomial patient bootstrap',
                      'across_splits': 'shared multinomial weights for union of test patients; fixed fitted models',
                      'primary_family': ['binary A BCE vs validation-selected clinical baseline',
                                         'binary B BCE vs validation-selected clinical baseline',
                                         'multi A focal vs validation-selected clinical baseline',
                                         'multi B focal vs validation-selected clinical baseline'],
                      'simultaneous_intervals': 'bootstrap max absolute standardized centered deviations for four contrasts'},
        'calibration_bins': 10, 'calibration_method': 'evaluate raw probabilities; no calibrator fitted',
        'lesion_aggregation': 'arithmetic mean probability per (patient_id, lesion_id); fixed threshold/argmax',
        'exploratory': True}


def probability(model, x, binary):
    p = model.predict_proba(x)
    assert np.array_equal(model.classes_, np.arange(2 if binary else 6))
    return p[:, 1] if binary else p


def save_tabular(parts, processor, name, task, seed, folder):
    binary = task == 'binary'
    xs = {s: processor.transform(p) for s, p in parts.items()}
    ys = {s: target(p, task) for s, p in parts.items()}
    weights = None if binary else compute_sample_weight('balanced', ys['train'])
    if name == 'metadata_logreg':
        candidates = [{'C': c} for c in PLAN['logistic_C']]
    elif name == 'metadata_boosting':
        candidates = PLAN['boosting_grid']
    else:
        candidates = [{'strategy': 'prior'}]
    history = []; best = -np.inf; selected = None; settings = None
    for params in candidates:
        if name == 'metadata_logreg':
            model = LogisticRegression(**params, penalty='l2', solver='lbfgs', max_iter=2000,
                                       random_state=seed, tol=1e-6)
        elif name == 'metadata_boosting':
            model = HistGradientBoostingClassifier(**params, **PLAN['boosting_fixed'], random_state=seed)
        else:
            model = DummyClassifier(**params, random_state=seed)
        model.fit(xs['train'], ys['train'], sample_weight=weights)
        if name == 'metadata_logreg' and np.max(model.n_iter_) >= 2000:
            raise ValueError('Logistic regression failed to converge')
        val = evaluate_probabilities(ys['val'], probability(model, xs['val'], binary), task)
        score = val['auc' if binary else 'balanced_accuracy']
        history.append({'parameters': params, 'selection_score': score, 'validation_metrics': val})
        if score > best:
            best = score; selected = model; settings = params
    folder.mkdir(parents=True, exist_ok=False)
    joblib.dump(selected, folder / 'model.joblib')
    cfg = {'seed': seed, 'task': task, 'model': name, 'selected_parameters': settings,
           'fixed_parameters': PLAN['boosting_fixed'] if name == 'metadata_boosting' else {},
           'best_validation_score': float(best), 'selection_metric': 'auc' if binary else 'balanced_accuracy',
           'preprocessor': processor.state, 'train_img_ids': parts['train'].img_id.tolist(),
           'val_img_ids': parts['val'].img_id.tolist(), 'multiclass_train_balanced_weights': not binary,
           'refit_train_plus_val': False}
    write_json(folder / 'configuration.json', cfg)
    write_json(folder / 'candidate_history.json', history)
    for split in ['val', 'test']:
        prob = probability(selected, xs[split], binary)
        prediction_frame(parts[split], prob, task).to_csv(folder / f'{split}_predictions.csv', index=False)
    met = evaluate_probabilities(ys['test'], probability(selected, xs['test'], binary), task)
    met.update(seed=seed, task=task, model=name, loss='log_loss' if name != 'metadata_prior' else 'prior',
               evaluated_split='test', smoke=False, best_validation_score=float(best))
    write_json(folder / 'metrics.json', met)
    print(f'{seed}/{task}/{name}: selected with validation from {len(candidates)} candidates', flush=True)
    return met


class ZeroMetadata:
    """Retains exact B capacity; clinical branch receives no clinical information."""
    def __init__(self, base):
        self.base = base; self.state = base.state
    def transform(self, frame):
        return np.zeros_like(self.base.transform(frame))


def main():
    import torch
    torch.set_num_threads(2)
    if DEST.exists():
        raise FileExistsError(f'Preserving existing run: {DEST}')
    df, config = verify(DEFAULT_DATA, DEFAULT_OUTPUT)
    visual, _ = embedding_index(DEFAULT_DATA, df)
    lookup = {img: i for i, img in enumerate(df.img_id)}
    DEST.mkdir(parents=True)
    write_json(DEST / 'design.json', PLAN)
    protected = [p for root in [DEFAULT_OUTPUT / 'splits', DEFAULT_OUTPUT / 'runs' / 'paper_baseline_v1',
                               DEFAULT_OUTPUT / 'runs' / 'metadata_only_v1'] for p in root.rglob('*') if p.is_file()]
    manifest = {'status': 'running', 'seeds': SEEDS, 'design_sha256': digest(DEST / 'design.json'),
                'protocol_sha256': digest(DEFAULT_OUTPUT / 'protocol.json'),
                'code_sha256': {p.name: digest(p) for p in [Path(__file__), Path(shared.__file__), Path(__file__).with_name('thesis_metrics.py')]},
                'protected_sha256': {str(p.resolve()): digest(p) for p in protected}, 'results': []}
    write_json(DEST / 'run_manifest.json', manifest)
    rows = []
    for seed in SEEDS:
        split = DEFAULT_OUTPUT / 'splits' / f'seed_{seed}'
        parts = {s: pd.read_csv(split / f'{s}_raw.csv') for s in ['train', 'val', 'test']}
        processor = TrainMetadata(json.loads((split / 'preprocessor.json').read_text()))
        for task in ['binary', 'multi']:
            for name in PLAN['clinical_models']:
                folder = DEST / f'seed_{seed}' / task / name
                rows.append(save_tabular(parts, processor, name, task, seed, folder))
            controls = [('B_metadata_capacity', 'focal', 'meta'), ('B_visual_capacity', 'focal', 'visual')]
            if task == 'binary':
                controls += [('A_focal', 'focal', 'all'), ('B_bce', 'bce', 'all'), ('metadata_mlp_focal', 'focal', 'meta_mlp')]
            for label, loss, inputs in controls:
                folder = DEST / f'seed_{seed}' / task / label
                if inputs == 'meta_mlp':
                    met = train_metadata(parts, processor, task, loss, seed, 50, 12, 'cpu', folder)
                else:
                    name = 'clip_dino_meta' if label == 'A_focal' else 'fusion'
                    features = {k: np.zeros_like(v) for k, v in visual.items()} if inputs == 'meta' else visual
                    preprocessing = ZeroMetadata(processor) if inputs == 'visual' else processor
                    met = shared.train_one(parts, preprocessing, features, lookup, task, name, loss,
                                           seed, 50, 12, 'cpu', folder)
                cfg = json.loads((folder / 'configuration.json').read_text())
                cfg.update(experiment_label=label, input_information=inputs,
                           capacity_control=inputs in ['meta', 'visual'],
                           masked_branches='constant zero inputs; bias parameters remain trainable')
                write_json(folder / 'configuration.json', cfg)
                met['model'] = label
                write_json(folder / 'metrics.json', met)
                rows.append(met)
                print(f'{seed}/{task}/{label}: completed', flush=True)
            pd.DataFrame(rows).to_csv(DEST / 'metrics_per_seed.csv', index=False)
    metrics = pd.DataFrame(rows)
    aggregate = metrics.groupby(['task', 'model', 'loss'])[['auc','accuracy','balanced_accuracy','precision','recall','f1']].agg(['mean', lambda x:x.std(ddof=0)])
    aggregate.columns = [f'{a}_{"mean" if b == "mean" else "std"}' for a,b in aggregate.columns]
    aggregate.to_csv(DEST / 'summary.csv')
    assert len(rows) == 65
    assert all(digest(Path(p)) == value for p,value in manifest['protected_sha256'].items())
    manifest.update(status='complete', completed_experiments=len(rows))
    write_json(DEST / 'run_manifest.json', manifest)
    print(f'COMPLETE: {len(rows)} new experiments; previous splits and results preserved.', flush=True)


if __name__ == '__main__':
    main()
