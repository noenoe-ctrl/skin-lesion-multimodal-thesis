"""Paired descriptive comparisons; preserves both original training runs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import DEFAULT_OUTPUT, ROOT, SEEDS, digest, write_json


def main():
    previous = DEFAULT_OUTPUT / 'runs' / 'paper_baseline_v1'
    current = DEFAULT_OUTPUT / 'runs' / 'metadata_only_v1'
    out = DEFAULT_OUTPUT / 'comparisons' / 'metadata_only_v1'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Comparison already exists: {out}')
    for folder in [previous, current]:
        assert json.loads((folder / 'results_verification.json').read_text())['status'] == 'passed'
    a = pd.read_csv(previous / 'metrics_per_seed.csv')
    m = pd.read_csv(current / 'metrics_per_seed.csv')
    assert len(a) == 90 and len(m) == 10
    assert set(m.seed) == set(SEEDS) and (m.evaluated_split == 'test').all()
    metrics = ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']
    summaries = pd.concat([pd.read_csv(folder / 'summary.csv') for folder in [previous, current]])
    pairs = []
    visual = {'clip_meta': 'clip', 'dino_meta': 'dino', 'convnext_meta': 'convnext',
              'clip_dino_meta': 'clip_dino', 'fusion': 'clip_dino'}
    for _, row in a.iterrows():
        baseline = m[(m.seed == row.seed) & (m.task == row.task)]
        assert len(baseline) == 1
        baseline = baseline.iloc[0]
        comparison = {'seed': int(row.seed), 'task': row.task, 'model': row.model,
                      'model_loss': row.loss, 'metadata_loss': baseline.loss,
                      'loss_matched': row.loss == baseline.loss,
                      'model_minus_metadata': {k: float(row[k] - baseline[k]) for k in metrics}}
        # Check exact test-image ordering before treating predictions as paired.
        p = previous / f'seed_{row.seed}' / row.task / f'{row.model}_{row.loss}' / 'test_predictions.csv'
        q = current / f'seed_{row.seed}' / row.task / f'metadata_mlp_{baseline.loss}' / 'test_predictions.csv'
        assert pd.read_csv(p).img_id.tolist() == pd.read_csv(q).img_id.tolist()
        if row.model in visual:
            v = a[(a.seed == row.seed) & (a.task == row.task) & (a.model == visual[row.model])].iloc[0]
            comparison['fusion_minus_visual'] = {k: float(row[k] - v[k]) for k in metrics}
            comparison['fusion_minus_best_unimodal'] = {
                k: float(row[k] - max(v[k], baseline[k])) for k in metrics}
        pairs.append(comparison)
    aggregates = []
    for task in ['binary', 'multi']:
        for model in a.model.unique():
            group = [p for p in pairs if p['task'] == task and p['model'] == model]
            assert len(group) == 5
            stats = {'task': task, 'model': model, 'seeds': SEEDS}
            for contrast in ['model_minus_metadata', 'fusion_minus_visual', 'fusion_minus_best_unimodal']:
                if contrast in group[0]:
                    stats[contrast] = {k: {'mean': float(np.mean([p[contrast][k] for p in group])),
                                          'std_ddof0': float(np.std([p[contrast][k] for p in group])),
                                          'positive_repetitions': sum(p[contrast][k] > 0 for p in group)} for k in metrics}
            aggregates.append(stats)
    sources = {str(p.resolve()): digest(p) for f in [previous, current]
               for p in [f / 'metrics_per_seed.csv', f / 'summary.csv', f / 'results_verification.json']}
    preservation = json.loads((DEFAULT_OUTPUT / 'metadata_extension_preservation.json').read_text())
    changed = [p for p, value in preservation['before'].items()
               if digest(Path(p) if Path(p).is_absolute() else ROOT / p) != value]
    assert not changed, changed
    report = {'status': 'passed', 'question': '¿Cuánto aportan los modelos fundacionales de imágenes a la clasificación de lesiones cutáneas cuando ya disponemos de metadata clínica, y qué ganancia adicional produce fusionar ambas fuentes?',
              'new_training_runs': 10, 'previous_training_runs': 90,
              'paired_test_image_ids_verified': True, 'existing_files_preserved': len(preservation['before']),
              'sources_sha256': sources, 'per_seed': pairs, 'aggregated_differences': aggregates,
              'statistical_significance_tested': False,
              'limitations': ['Repeated holdouts are dependent.', 'Binary B uses focal; A and metadata use BCE.',
                              'Metadata MLP is the only completed clinical baseline.']}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'comparisons.json', report)
    labels = {'metadata_mlp':'Metadata sola (MLP)', 'clip':'CLIP', 'dino':'DINOv2', 'convnext':'ConvNeXt',
              'clip_dino':'CLIP + DINOv2', 'clip_meta':'CLIP + metadata', 'dino_meta':'DINOv2 + metadata',
              'convnext_meta':'ConvNeXt + metadata', 'clip_dino_meta':'A: CLIP + DINOv2 + metadata', 'fusion':'B: fusión intermedia'}
    order = ['metadata_mlp','clip','dino','convnext','clip_dino','clip_meta','dino_meta','convnext_meta','clip_dino_meta','fusion']
    text = ['# Resultados completos: metadata sola y aporte incremental', '',
            report['question'], '',
            'Evaluación: PAD-UFES-20, cinco holdouts agrupados por paciente, selección con validación y preprocesamiento ajustado solo en train.', '',
            'Completados y verificados: 10 entrenamientos nuevos de metadata MLP, comparables con los 90 existentes. Los archivos previos permanecen intactos.', '',
            'Todas las tablas muestran media ± desviación poblacional (ddof=0), no intervalos de confianza. Son diferencias descriptivas sin prueba de significancia.', '']
    for task in ['binary', 'multi']:
        text += [f'## {"Binaria" if task == "binary" else "Multiclase"}', '',
                 '| Modelo | AUC / macro OvR | Accuracy | BACC | Recall | F1 |',
                 '|---|---:|---:|---:|---:|---:|']
        for model in order:
            row = summaries[(summaries.task == task) & (summaries.model == model)].iloc[0]
            values = [f'{row[k+"_mean"]:.4f} ± {row[k+"_std"]:.4f}' for k in ['auc','accuracy','balanced_accuracy','recall','f1']]
            text += ['| ' + labels[model] + ' | ' + ' | '.join(values) + ' |']
        primary = 'auc' if task == 'binary' else 'balanced_accuracy'
        text += ['', f'Ganancia de fusión sobre metadata: {"AUC" if task == "binary" else "BACC"}, en puntos porcentuales.', '',
                 '| Fusión | Diferencia media | Repeticiones positivas / 5 |', '|---|---:|---:|']
        for model in visual:
            stat = next(s for s in aggregates if s['task'] == task and s['model'] == model)['model_minus_metadata'][primary]
            text += [f'| {labels[model]} | {100*stat["mean"]:+.2f} | {stat["positive_repetitions"]} |']
        text += ['']
    text += ['## Interpretación y pendiente', '',
             'En binario, metadata MLP tiene mayor AUC media que las cuatro configuraciones visuales solas. A y B tienen mayor AUC media que metadata, pero la ganancia es menor que frente a imágenes solas. En multiclase, A y B mejoran la BACC media frente a metadata. Esto no demuestra significancia ni superioridad general de los modelos fundacionales.', '',
             'B binario usa focal mientras metadata y A usan BCE: su diferencia mezcla pérdida y arquitectura. Quedan los controles de pérdida/capacidad, regresión logística y boosting clínicos, análisis de incertidumbre por paciente y calibración. No fueron ejecutados en esta extensión.', '',
             'El archivo comparisons.json conserva las diferencias de cada semilla, sus agregados y hashes de las fuentes. Las predicciones, checkpoints, historiales y métricas nuevas están en artifacts/runs/metadata_only_v1.', '']
    (out / 'RESULTADOS_METADATA_COMPLETOS.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'VERIFIED paired comparisons; {len(pairs)} pairs; {len(preservation["before"])} existing files unchanged. Report: {out}')


if __name__ == '__main__':
    main()
