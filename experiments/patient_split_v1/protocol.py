"""PAD-UFES-20: patient-grouped splits and train-only metadata preprocessing.

This module never trains a model and never writes in the historical dataset.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ['BCC', 'MEL', 'SCC', 'ACK', 'NEV', 'SEK']
SEEDS = [42, 123, 456, 789, 2024]
NUMERIC = ['age', 'fitspatrick', 'diameter_1', 'diameter_2']
META = ['smoke', 'drink', 'pesticide', 'gender', 'skin_cancer_history',
        'cancer_history', 'has_piped_water', 'has_sewage_system', *NUMERIC,
        'region', 'background_father', 'background_mother', 'itch', 'grew',
        'hurt', 'changed', 'bleed', 'elevation']
CATEGORICAL = [c for c in META if c not in NUMERIC]
UNKNOWN = '__UNKNOWN__'
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'dataset' / 'PAD-UFES-20'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'artifacts'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def historical_hashes(data):
    paths = list(data.parent.glob('*.ipynb')) + list(data.glob('*.csv'))
    paths += [p for p in (data / 'resultados').rglob('*') if p.is_file()]
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def load_raw(data):
    df = pd.read_csv(Path(data) / 'metadata.csv', dtype={'patient_id': str, 'img_id': str})
    required = {'patient_id', 'lesion_id', 'img_id', 'diagnostic', *META}
    if not required <= set(df):
        raise ValueError(f'Missing columns: {required - set(df)}')
    if df[['patient_id', 'lesion_id', 'img_id', 'diagnostic']].isna().any().any():
        raise ValueError('Missing identifiers or labels')
    if df.img_id.duplicated().any() or set(df.diagnostic) != set(CLASSES):
        raise ValueError('Duplicate images or unexpected classes')
    if df.groupby(['patient_id', 'lesion_id']).diagnostic.nunique().max() != 1:
        raise ValueError('Conflicting diagnoses within a patient/lesion pair')
    df = df.sort_values('img_id').reset_index(drop=True)
    df['label'] = df.diagnostic.isin(['BCC', 'MEL', 'SCC']).astype(int)
    return df


def normalize_metadata(df):
    """Only fixed per-value cleanup; no fitted statistics or label access."""
    out = df[META].copy()
    for c in NUMERIC:
        s = out[c].replace(r'^\s*(UNK|UNKNOWN)?\s*$', np.nan, regex=True)
        out[c] = pd.to_numeric(s, errors='raise')
    for c in CATEGORICAL:
        s = out[c].astype('string').str.strip().str.upper()
        out[c] = s.mask(s.isin(['', 'UNK', 'UNKNOWN']))
    return out


class TrainMetadata:
    """Serializable median/mode, min-max and vocabulary fitted ONLY on train.

    Statistics are image-row weighted, matching the supervised sample unit.
    No patient_id, lesion_id, diagnostic, biopsed or label is ever read by fit.
    """
    def __init__(self, state=None):
        self.state = state

    def fit(self, train):
        x = normalize_metadata(train)
        state = {'policy': 'train-only median/mode; UNK=missing; minmax by name',
                 'fit_rows': len(train), 'numeric': {}, 'categorical': {}}
        for c in NUMERIC:
            if x[c].notna().sum() == 0:
                raise ValueError(f'{c}: entirely missing in train; explicit policy needed')
            median = float(x[c].median())
            filled = x[c].fillna(median)
            state['numeric'][c] = {'median': median, 'min': float(filled.min()),
                                   'max': float(filled.max())}
        for c in CATEGORICAL:
            counts = x[c].dropna().value_counts()
            mode = sorted(counts[counts == counts.max()].index)[0] if len(counts) else UNKNOWN
            vocab = sorted(set(x[c].dropna().tolist()) | {mode, UNKNOWN})
            state['categorical'][c] = {'mode': mode, 'categories': vocab}
        state['feature_names'] = NUMERIC + [f'{c}={v}' for c in CATEGORICAL
                                            for v in state['categorical'][c]['categories']]
        self.state = state
        return self

    def impute(self, df):
        x = normalize_metadata(df)
        for c in NUMERIC:
            x[c] = x[c].fillna(self.state['numeric'][c]['median'])
        for c in CATEGORICAL:
            x[c] = x[c].fillna(self.state['categorical'][c]['mode'])
        return x

    def transform(self, df):
        x = self.impute(df)
        columns = []
        for c in NUMERIC:
            s = self.state['numeric'][c]
            columns.append(((x[c] - s['min']) / (s['max'] - s['min'] or 1)).to_numpy(float))
        for c in CATEGORICAL:
            vocab = self.state['categorical'][c]['categories']
            vals = x[c].where(x[c].isin(vocab), UNKNOWN)
            columns.extend((vals == v).to_numpy(dtype=float) for v in vocab)
        a = np.column_stack(columns).astype(np.float32)
        if not np.isfinite(a).all():
            raise ValueError('Non-finite metadata features')
        return a


def create_split(df, seed):
    """Ten grouped strata, block 0=test, 1=val, remaining 8=train.

    Randomly recode group IDs to break greedy ties reproducibly. shuffle=False
    avoids depending on group-shuffling implementations in sklearn versions.
    No selection based on model scores, no patient majority-label reduction.
    """
    patients = np.array(sorted(df.patient_id.unique()))
    order = np.random.default_rng(seed).permutation(patients)
    codes = {p: i for i, p in enumerate(order)}
    groups = df.patient_id.map(codes).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=False)
    blocks = [test for _, test in splitter.split(np.zeros((len(df), 1)), df.diagnostic, groups)]
    assigned = np.full(len(df), 'train', dtype=object)
    assigned[blocks[0]] = 'test'
    assigned[blocks[1]] = 'val'
    return {s: df.loc[assigned == s].reset_index(drop=True) for s in ['train', 'val', 'test']}


def audit_split(df, parts):
    assert set(parts) == {'train', 'val', 'test'}
    combined = pd.concat(parts.values(), ignore_index=True)
    assert len(combined) == len(df) and not combined.img_id.duplicated().any()
    assert set(combined.img_id) == set(df.img_id)
    original = df.set_index('img_id')
    aligned = original.loc[combined.img_id].reset_index()
    a = combined[original.columns].reset_index(drop=True).copy()
    b = aligned[original.columns].reset_index(drop=True).copy()
    # CSV parsing may infer bool in subsets without UNK, and str elsewhere.
    # Compare the same values canonically without treating UNK as missing here.
    for c in CATEGORICAL + ['biopsed']:
        if c in a:
            a[c] = a[c].astype('string').str.upper()
            b[c] = b[c].astype('string').str.upper()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    intersections = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        counts = {'patients': len(set(parts[a].patient_id) & set(parts[b].patient_id)),
                  'lesions': len(set(zip(parts[a].patient_id, parts[a].lesion_id)) &
                                 set(zip(parts[b].patient_id, parts[b].lesion_id))),
                  'images': len(set(parts[a].img_id) & set(parts[b].img_id))}
        assert not any(counts.values()), counts
        intersections[f'{a}-{b}'] = counts
    rows = []
    global_counts = df.diagnostic.value_counts()
    for split, p in parts.items():
        assert set(p.diagnostic) == set(CLASSES), f'{split}: missing class'
        row = {'split': split, 'images': len(p), 'patients': p.patient_id.nunique(),
               'lesions': len(p[['patient_id', 'lesion_id']].drop_duplicates()),
               'image_fraction': len(p) / len(df)}
        row.update({c: int((p.diagnostic == c).sum()) for c in CLASSES})
        row['max_class_proportion_deviation'] = max(abs(row[c] / len(p) - global_counts[c] / len(df))
                                                   for c in CLASSES)
        rows.append(row)
    return rows, intersections


def embedding_index(data, df):
    """Use old split CSVs ONLY as an image/order index, never as metadata."""
    old = [pd.read_csv(data / f'split_{s}.csv') for s in ['train', 'val', 'test']]
    ids = pd.concat(old).img_id.tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(df.img_id):
        raise ValueError('Historical cache does not cover original images uniquely')
    labels = df.set_index('img_id').diagnostic
    hashes = {}
    arrays = {}
    for model, dim in [('clip', 768), ('dino', 1024), ('convnext', 1024)]:
        chunks = []
        for s, o in zip(['train', 'val', 'test'], old):
            if not o.diagnostic.reset_index(drop=True).equals(labels.loc[o.img_id].reset_index(drop=True)):
                raise ValueError('Historical labels do not match original metadata')
            path = data / 'resultados' / f'{model}_X_{s}.npy'
            x = np.load(path, allow_pickle=False)
            if x.shape != (len(o), dim) or not np.isfinite(x).all():
                raise ValueError(f'Invalid cache: {path}')
            chunks.append(x)
            hashes[str(path.resolve())] = digest(path)
        lookup = {img: i for i, img in enumerate(ids)}
        arrays[model] = np.concatenate(chunks)[[lookup[img] for img in df.img_id]].astype(np.float32)
    return arrays, hashes


def prepare(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output == data or data in output.parents:
        raise ValueError('Outputs must be outside historical PAD-UFES-20 directory')
    if output.exists():
        raise FileExistsError(f'Preserving existing artifacts: {output}; use verify or a new directory')
    before = historical_hashes(data)
    df = load_raw(data)
    _, cache_hashes = embedding_index(data, df)
    # Validate everything in memory before creating any artifacts.
    prepared = []
    for seed in SEEDS:
        parts = create_split(df, seed)
        rows, overlaps = audit_split(df, parts)
        processor = TrainMetadata().fit(parts['train'])
        matrices = {s: processor.transform(p) for s, p in parts.items()}
        prepared.append((seed, parts, rows, overlaps, processor, matrices))
    output.mkdir(parents=True)
    summary = []
    for seed, parts, rows, overlaps, processor, matrices in prepared:
        folder = output / 'splits' / f'seed_{seed}'
        folder.mkdir(parents=True)
        manifest = []
        for s, p in parts.items():
            p.to_csv(folder / f'{s}_raw.csv', index=False)
            imputed = p.copy()
            imputed[META] = processor.impute(p)
            imputed.to_csv(folder / f'{s}_imputed.csv', index=False)
            np.save(folder / f'{s}_metadata.npy', matrices[s])
            manifest.append(p[['img_id', 'patient_id', 'lesion_id', 'diagnostic', 'label']].assign(split=s))
        pd.concat(manifest).to_csv(folder / 'manifest.csv', index=False)
        write_json(folder / 'preprocessor.json', processor.state)
        write_json(folder / 'audit.json', {'overlaps': overlaps, 'summary': rows})
        summary.extend(dict(seed=seed, **row) for row in rows)
    pd.DataFrame(summary).to_csv(output / 'split_summary.csv', index=False)
    write_json(output / 'historical_sha256.json', before)
    write_json(output / 'protocol.json', {
        'version': 'patient_split_v1', 'data_dir': str(data),
        'metadata_sha256': digest(data / 'metadata.csv'), 'classes': CLASSES,
        'seeds': SEEDS, 'unit': 'image', 'group': 'patient_id',
        'design': '5 repeated grouped holdouts; 10 stratified blocks, 0=test, 1=val',
        'target_fractions': {'train': .8, 'val': .1, 'test': .1},
        'group_ties': 'seeded patient recoding; StratifiedGroupKFold shuffle=False',
        'preprocessing': 'train-only image-row medians/modes; UNK missing; named minmax; train vocab',
        'binary_losses': 'BCE for concat heads; focal(alpha=.25,gamma=2) for intermediate fusion',
        'multiclass_loss': 'standard focal: class weight outside unweighted probability factor',
        'historical_difference': 'categoricals all one-hot; explicit unseen category; corrected focal math',
        'embedding_hashes': cache_hashes,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__}})
    verify(data, output)
    return pd.DataFrame(summary)


def verify(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    config = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    assert digest(data / 'metadata.csv') == config['metadata_sha256']
    df = load_raw(data)
    summaries = []
    for seed in config['seeds']:
        folder = output / 'splits' / f'seed_{seed}'
        parts = {s: pd.read_csv(folder / f'{s}_raw.csv') for s in ['train', 'val', 'test']}
        rows, overlaps = audit_split(df, parts)
        expected_manifest = pd.concat([
            p[['img_id', 'patient_id', 'lesion_id', 'diagnostic', 'label']].assign(split=s)
            for s, p in parts.items()], ignore_index=True)
        pd.testing.assert_frame_equal(pd.read_csv(folder / 'manifest.csv'), expected_manifest,
                                      check_dtype=False)
        assert json.loads((folder / 'audit.json').read_text(encoding='utf-8')) == {
            'overlaps': overlaps, 'summary': rows}
        regenerated = create_split(df, seed)
        for s in parts:
            assert parts[s].img_id.tolist() == regenerated[s].img_id.tolist(), 'Non-reproducible split'
        processor = TrainMetadata(json.loads((folder / 'preprocessor.json').read_text(encoding='utf-8')))
        assert processor.state == TrainMetadata().fit(parts['train']).state, 'Fit is not train-only'
        for s, p in parts.items():
            np.testing.assert_array_equal(processor.transform(p), np.load(folder / f'{s}_metadata.npy'))
            saved = pd.read_csv(folder / f'{s}_imputed.csv')
            actual = normalize_metadata(saved)
            pd.testing.assert_frame_equal(actual, processor.impute(p), check_dtype=False)
        # Changing labels must have no influence on fitted or transformed metadata.
        poison = parts['train'].copy(); poison['diagnostic'] = 'NOT_A_CLASS'; poison['label'] = -999
        assert TrainMetadata().fit(poison).state == processor.state
        # Validation/test values and categories cannot alter train statistics.
        for s in ['val', 'test']:
            probe = parts[s].copy()
            probe[NUMERIC] = 1e9; probe[CATEGORICAL] = 'NEVER_SEEN_CATEGORY'
            assert np.isfinite(processor.transform(probe)).all()
        summaries.extend(dict(seed=seed, **row) for row in rows)
    recorded = pd.read_csv(output / 'split_summary.csv')
    pd.testing.assert_frame_equal(recorded, pd.DataFrame(summaries), check_dtype=False, atol=1e-12)
    for path, h in json.loads((output / 'historical_sha256.json').read_text(encoding='utf-8')).items():
        assert digest(path) == h, f'Historical file changed: {path}'
    for path, h in config['embedding_hashes'].items():
        assert digest(path) == h
    print('VERIFIED: zero shared patients/lesions/images; complete coverage; all six classes;')
    print('reproducible splits; train-only statistics; label independence; finite features; historical files unchanged.')
    return df, config
