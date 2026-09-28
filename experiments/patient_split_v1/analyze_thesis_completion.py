"""Audited all-model comparisons and patient-cluster uncertainty on fixed fits."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from protocol import CLASSES, DEFAULT_OUTPUT, SEEDS, TrainMetadata, digest, write_json
from thesis_metrics import evaluate_probabilities, probabilities, target, weighted_primary

OUT = DEFAULT_OUTPUT / 'comparisons' / 'thesis_completion_v2'
RUNS = ['paper_baseline_v1','metadata_only_v1','thesis_completion_v1','metadata_empirical_prior_v1']
BOOT = 2000
METRICS = ['auc','accuracy','balanced_accuracy','precision','recall','f1','average_precision','brier','ece_10_bins']
PRIMARY = {'binary':'auc','multi':'balanced_accuracy'}
LABELS = {'metadata_mlp':'Metadata MLP','metadata_logreg':'Metadata logística','metadata_boosting':'Metadata boosting',
          'metadata_prior':'Prior ponderado en multiclase','metadata_empirical_prior':'Prior empírico de train',
          'clip':'CLIP','dino':'DINOv2','convnext':'ConvNeXt','clip_dino':'CLIP + DINOv2',
          'clip_meta':'CLIP + metadata','dino_meta':'DINOv2 + metadata','convnext_meta':'ConvNeXt + metadata',
          'clip_dino_meta':'A (BCE binaria)','fusion':'B (focal binaria)',
          'A_focal':'A focal','B_bce':'B BCE','metadata_mlp_focal':'Metadata MLP focal',
          'B_metadata_capacity':'B solo metadata, misma capacidad','B_visual_capacity':'B solo imágenes, misma capacidad',
          'metadata_selected':'Metadata elegida con validación'}


def load_records():
    records = {task:{seed:{} for seed in SEEDS} for task in ['binary','multi']}
    sources = {}; count = 0
    for name in RUNS:
        folder = DEFAULT_OUTPUT / 'runs' / name
        manifest = json.loads((folder/'run_manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['protocol_sha256'] == digest(DEFAULT_OUTPUT/'protocol.json')
        if 'protected_sha256' in manifest:
            assert all(digest(Path(p)) == h for p,h in manifest['protected_sha256'].items())
        table = pd.read_csv(folder/'metrics_per_seed.csv')
        assert len(table) == manifest['completed_experiments']
        for _, row in table.iterrows():
            seed = int(row.seed); task = row.task
            suffix = f'{row.model}_{row.loss}' if name in RUNS[:2] else row.model
            model_dir = folder/f'seed_{seed}'/task/suffix
            pred = pd.read_csv(model_dir/'test_predictions.csv')
            split_dir = DEFAULT_OUTPUT/'splits'/f'seed_{seed}'
            raw = pd.read_csv(split_dir/'test_raw.csv')
            for col in ['img_id','patient_id','lesion_id','diagnostic']:
                assert pred[col].tolist() == raw[col].tolist()
            truth = target(raw,task); np.testing.assert_array_equal(pred.target,truth)
            prob = probabilities(pred,task)
            point = evaluate_probabilities(truth,prob,task)
            for metric in ['auc','accuracy','balanced_accuracy','precision','recall','f1']:
                np.testing.assert_allclose(row[metric],point[metric],rtol=0,atol=1e-8)
            expected_pred = (prob>=.5).astype(int) if task=='binary' else prob.argmax(axis=1)
            np.testing.assert_array_equal(pred.prediction,expected_pred)
            cfg = json.loads((model_dir/'configuration.json').read_text())
            state = json.loads((split_dir/'preprocessor.json').read_text())
            assert cfg['preprocessor'] == state
            for split in ['train','val']:
                assert cfg[f'{split}_img_ids'] == pd.read_csv(split_dir/f'{split}_raw.csv').img_id.tolist()
            if (model_dir/'history.csv').exists():
                hist = pd.read_csv(model_dir/'history.csv')
                assert row.best_epoch == hist.loc[hist.selection_score.idxmax(),'epoch']
                np.testing.assert_allclose(row.best_validation_score,hist.selection_score.max(),atol=1e-12)
            else:
                clf = joblib.load(model_dir/'model.joblib')
                for split in ['val','test']:
                    raw_split = pd.read_csv(split_dir/f'{split}_raw.csv')
                    actual = clf.predict_proba(TrainMetadata(state).transform(raw_split))
                    actual = actual[:,1] if task=='binary' else actual
                    saved = pd.read_csv(model_dir/f'{split}_predictions.csv')
                    assert saved.img_id.tolist() == raw_split.img_id.tolist()
                    np.testing.assert_allclose(actual,probabilities(saved,task),atol=1e-12)
                    if split == 'val':
                        val = evaluate_probabilities(target(raw_split,task),actual,task)
                        np.testing.assert_allclose(row.best_validation_score,val[PRIMARY[task]],atol=1e-12)
                if (model_dir/'candidate_history.json').exists():
                    hist = json.loads((model_dir/'candidate_history.json').read_text())
                    np.testing.assert_allclose(row.best_validation_score,max(c['selection_score'] for c in hist),atol=1e-12)
            grouped = pred.groupby(['patient_id','lesion_id'],sort=True)
            assert grouped.target.nunique().max() == 1
            cols = ['prob_malignant'] if task=='binary' else [f'prob_{c}' for c in CLASSES]
            lesions = grouped[cols].mean()
            lp = lesions.prob_malignant.to_numpy() if task=='binary' else lesions[cols].to_numpy()
            lesion_metrics = evaluate_probabilities(grouped.target.first().to_numpy(dtype=int),lp,task)
            sources[str((model_dir/'test_predictions.csv').resolve())] = digest(model_dir/'test_predictions.csv')
            records[task][seed][row.model] = {'frame':pred,'prob':prob,'point':point,
                'lesion_metrics':lesion_metrics,'validation_score':float(row.best_validation_score),
                'config':cfg,'run':name,'loss':row.loss,'folder':model_dir}
            count += 1
        # Preserve a verification report separately: original run artifacts stay untouched.
    assert count == 175
    selectors = []
    for task in records:
        for seed, models in records[task].items():
            choices = ['metadata_mlp','metadata_logreg','metadata_boosting']
            winner = max(choices,key=lambda model:models[model]['validation_score'])
            models['metadata_selected'] = models[winner]
            selectors.append({'seed':seed,'task':task,'selected_model':winner,
                              'validation_scores':{c:models[c]['validation_score'] for c in choices}})
            b = models['fusion']['config']['parameter_count']
            assert models['B_metadata_capacity']['config']['parameter_count'] == b
            assert models['B_visual_capacity']['config']['parameter_count'] == b
    return records,sources,selectors,count


def cluster_weights(frames,draws,rng):
    """Same patient multiplicity across all appearances, models, tasks and repeats."""
    patients = sorted(set().union(*(set(frame.patient_id) for frame in frames)))
    lookup = {p:i for i,p in enumerate(patients)}
    indices = [np.array([lookup[p] for p in frame.patient_id]) for frame in frames]
    truths = [frame.diagnostic.to_numpy() for frame in frames]
    batches = []; accepted = 0; rejected = 0
    while accepted < draws:
        weights = rng.multinomial(len(patients),np.full(len(patients),1/len(patients)),size=draws)
        good = np.ones(draws,dtype=bool)
        for idx,y in zip(indices,truths):
            local = weights[:,idx]
            for c in CLASSES:
                good &= local[:,y==c].sum(axis=1)>0
        rejected += int((~good).sum())
        chosen = weights[good][:draws-accepted]
        batches.append(chosen); accepted += len(chosen)
        if not good.any():
            raise ValueError('Bootstrap cannot retain all classes')
    global_weights = np.concatenate(batches)
    return [global_weights[:,idx].astype(float) for idx in indices],rejected,len(patients)


def interval(values):
    return [float(v) for v in np.quantile(values,[.025,.975])]


def contrasts(task,models):
    result = [(m,'metadata_selected','vs_clinical_selected') for m in models if not m.startswith('metadata') and not m.startswith('B_')]
    visual = {'clip_meta':'clip','dino_meta':'dino','convnext_meta':'convnext','clip_dino_meta':'clip_dino',
              'fusion':'clip_dino','A_focal':'clip_dino','B_bce':'clip_dino'}
    result += [(m,v,'vs_visual') for m,v in visual.items() if m in models]
    result += [('fusion','B_metadata_capacity','B_vs_metadata_same_capacity'),
               ('fusion','B_visual_capacity','B_vs_visual_same_capacity')]
    if task == 'binary':
        result += [('B_bce','clip_dino_meta','B_vs_A_BCE'),('fusion','A_focal','B_vs_A_focal'),
                   ('A_focal','clip_dino_meta','A_focal_vs_BCE'),('fusion','B_bce','B_focal_vs_BCE'),
                   ('metadata_mlp_focal','metadata_mlp','metadata_focal_vs_BCE'),
                   ('B_bce','metadata_selected','vs_clinical_selected')]
    else:
        result += [('fusion','clip_dino_meta','B_vs_A_focal')]
    # Exploratory contrasts against each clinical reference, not only the selector.
    fusion_models = ['clip_meta','dino_meta','convnext_meta','clip_dino_meta','fusion','A_focal','B_bce']
    result += [(m,c,'vs_individual_clinical') for m in fusion_models if m in models
               for c in ['metadata_mlp','metadata_logreg','metadata_boosting']]
    return list(dict.fromkeys(result))


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving existing analysis: {OUT}')
    records,sources,selectors,count = load_records()
    rng = np.random.default_rng(20260928)
    first = [records['multi'][seed]['clip']['frame'] for seed in SEEDS]
    weights,rejected,patients = cluster_weights(first,BOOT,rng)
    aggregate = []; paired = []; local_intervals = []; curves = []; primary_draws = {}
    for task in ['binary','multi']:
        names = list(records[task][SEEDS[0]])
        draws = {}; points = {}
        for name in names:
            means = []
            for i,seed in enumerate(SEEDS):
                r = records[task][seed][name]
                means.append(weighted_primary(r['frame'].target.to_numpy(),r['prob'],task,weights[i]))
                curves.append({'seed':seed,'task':task,'model':name,'calibration':r['point']['calibration_bins'],
                               'definition':r['point']['calibration_definition'],
                               'confusion_matrix':r['point']['confusion_matrix'],'class_recall':r['point']['class_recall']})
            draws[name] = np.mean(means,axis=0)
            p = [records[task][s][name]['point'] for s in SEEDS]
            points[name] = float(np.mean([v[PRIMARY[task]] for v in p]))
            summary = {'task':task,'model':name,'primary_metric':PRIMARY[task],
                       'primary_ci95_shared_patient_bootstrap':interval(draws[name]),
                       'metrics':{k:{'mean':float(np.mean([v[k] for v in p])),
                                     'std_ddof0':float(np.std([v[k] for v in p]))} for k in METRICS},
                       'lesion_sensitivity':{k:{'mean':float(np.mean([records[task][s][name]['lesion_metrics'][k] for s in SEEDS])),
                                               'std_ddof0':float(np.std([records[task][s][name]['lesion_metrics'][k] for s in SEEDS]))} for k in METRICS}}
            aggregate.append(summary)
        for seed in SEEDS:
            local,_rej,_n = cluster_weights([records[task][seed]['clip']['frame']],BOOT,rng)
            d = {name:weighted_primary(records[task][seed][name]['frame'].target.to_numpy(),records[task][seed][name]['prob'],task,local[0]) for name in names}
            for left,right,kind in contrasts(task,names):
                local_intervals.append({'seed':seed,'task':task,'left':left,'right':right,'contrast':kind,
                    'difference':records[task][seed][left]['point'][PRIMARY[task]]-records[task][seed][right]['point'][PRIMARY[task]],
                    'ci95':interval(d[left]-d[right])})
        for left,right,kind in contrasts(task,names):
            difference = draws[left]-draws[right]
            observed = points[left]-points[right]
            per_seed = [records[task][s][left]['point'][PRIMARY[task]]-records[task][s][right]['point'][PRIMARY[task]] for s in SEEDS]
            entry = {'task':task,'left':left,'right':right,'contrast':kind,'metric':PRIMARY[task],
                     'mean_difference':observed,'ci95_shared_patient':interval(difference),
                     'per_seed_difference':dict(zip(map(str,SEEDS),map(float,per_seed))),
                     'positive_repetitions':sum(v>0 for v in per_seed), 'simultaneous_primary_ci95':None}
            paired.append(entry)
            if right=='metadata_selected' and left in (['clip_dino_meta','B_bce'] if task=='binary' else ['clip_dino_meta','fusion']):
                primary_draws[(task,left)] = (entry,difference)
    assert len(primary_draws)==4
    standardized = []
    for entry,d in primary_draws.values():
        sd = float(np.std(d,ddof=1)); assert sd>0
        standardized.append(np.abs((d-entry['mean_difference'])/sd))
    critical = float(np.quantile(np.max(standardized,axis=0),.95))
    for entry,d in primary_draws.values():
        margin = critical*float(np.std(d,ddof=1))
        entry['simultaneous_primary_ci95'] = [entry['mean_difference']-margin,entry['mean_difference']+margin]
    report = {'status':'passed','verified_training_runs':count,'new_training_runs':75,
              'clinical_selection':selectors,'all_models':aggregate,'paired_contrasts':paired,
              'within_split_paired_ci':local_intervals,'bootstrap':{'draws':BOOT,'seed':20260928,
              'test_patient_union':patients,'rejected_draws_shared':rejected,'cluster':'patient_id',
              'shared_patient_weights_across_repetitions':True,'primary_family_size':4,'max_standardized_critical':critical,
              'conditional_on_fixed_trained_models':True,
              'limitations':'Does not capture training-set overlap/retraining uncertainty or clinical-baseline selection uncertainty. Exploratory intervals; not confirmatory significance.'},
              'prediction_sha256':sources,'raw_calibration':curves,
              'analysis_code_sha256':digest(Path(__file__)),
              'individual_clinical_contrasts':'Exploratory secondary contrasts, including boosting/logistic separately.'}
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'analysis.json',report)
    np.savez_compressed(OUT/'primary_bootstrap.npz',**{f'{task}_{model}':d for (task,model),(_e,d) in primary_draws.items()})
    write_report(report)
    print(f'VERIFIED {count} fits; patient-cluster intervals and all-model analysis complete: {OUT}',flush=True)


def write_report(report):
    lines = ['# Investigación de tesis: comparación completa', '',
             '¿Cuánto aportan los modelos fundacionales de imágenes cuando ya disponemos de metadata clínica, y qué ganancia adicional produce fusionar ambas fuentes?', '',
             'Evaluación en PAD-UFES-20: mismos cinco holdouts por paciente, estadísticas solo de train y selección con validación. Se conservaron los 100 entrenamientos previos y se completaron 75 referencias/controles nuevos (65 en thesis_completion_v1 y 10 priors empíricos).', '',
             '## Qué debe compararse', '',
             'Se incluyen todos los modelos. A y B son contrastes principales; los encoders individuales, sus fusiones y las referencias clínicas explican el origen de la ganancia. La referencia clínica principal se elige entre MLP, logística y boosting con validación en cada semilla/tarea; nunca por resultados de test.', '',
             'El prior ponderado usa pesos balanceados en multiclase (por ello es uniforme); el prior empírico adicional usa las frecuencias originales de train sin ponderar.', '',
             '## Selección clínica por validación', '', '| Semilla | Tarea | Modelo elegido |', '|---|---|---|']
    for s in report['clinical_selection']:
        lines.append(f'| {s["seed"]} | {s["task"]} | {LABELS[s["selected_model"]]} |')
    for task in ['binary','multi']:
        lines += ['',f'## {"Binaria" if task=="binary" else "Multiclase"}: todos los modelos', '',
                  'Media ± desviación poblacional; umbral 0.5/argmax. AP=average precision. Brier multiclase=sumatoria por clase, sin dividir por seis.', '',
                  '| Modelo | AUC/OvR | BACC | F1 | AP | Brier | ECE |', '|---|---:|---:|---:|---:|---:|---:|']
        for r in report['all_models']:
            if r['task']==task:
                vals = [f'{r["metrics"][k]["mean"]:.4f} ± {r["metrics"][k]["std_ddof0"]:.4f}' for k in ['auc','balanced_accuracy','f1','average_precision','brier','ece_10_bins']]
                lines.append('| '+LABELS[r['model']]+' | '+' | '.join(vals)+' |')
        lines += ['', '### Contrastes principales frente a la referencia clínica elegida', '',
                  'Diferencias e intervalos en puntos porcentuales. IC simultáneo: familia de cuatro contrastes; bootstrap por paciente con pesos compartidos entre repeticiones.', '',
                  '| Comparación | Ganancia media | IC95 simultáneo | Repeticiones positivas / 5 |', '|---|---:|---:|---:|']
        for p in report['paired_contrasts']:
            if p['task']==task and p['simultaneous_primary_ci95'] is not None:
                lo,hi=p['simultaneous_primary_ci95']
                lines.append(f'| {LABELS[p["left"]]} − clínica elegida | {100*p["mean_difference"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {p["positive_repetitions"]} |')
        lines += ['', '### Controles de pérdida y capacidad', '',
                  '| Contraste | Diferencia media | IC95 por paciente (sin ajuste de multiplicidad) |', '|---|---:|---:|']
        for p in report['paired_contrasts']:
            if p['task']==task and p['contrast'] not in ['vs_clinical_selected','vs_visual']:
                lo,hi=p['ci95_shared_patient']
                lines.append(f'| {LABELS[p["left"]]} − {LABELS[p["right"]]} | {100*p["mean_difference"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
        lines += ['', '### Sensibilidad al evaluar por lesión', '',
                  'Promedio aritmético de probabilidades de sus imágenes, agrupando (patient_id, lesion_id), sin seleccionar la regla con test.', '',
                  '| Modelo | Métrica principal por imagen | Métrica principal por lesión |', '|---|---:|---:|']
        for r in report['all_models']:
            if r['task']==task and r['model'] in ['metadata_selected','clip','dino','convnext','clip_dino','clip_dino_meta','fusion','B_bce']:
                metric=PRIMARY[task]
                lines.append(f'| {LABELS[r["model"]]} | {r["metrics"][metric]["mean"]:.4f} | {r["lesion_sensitivity"][metric]["mean"]:.4f} |')
    lines += ['', '## Verificación, incertidumbre y límites', '',
              'Además del selector clínico, analysis.json conserva contrastes exploratorios contra MLP, logística y boosting individualmente. Elegir por validación puede no coincidir con el mejor score de test: no debe presentarse al selector como el mejor modelo de test ni ocultarse el desempeño de las referencias individuales.', '',
              'Se recalcularon seis métricas desde todas las predicciones; IDs/etiquetas y preprocesadores coinciden con los mismos splits. Los modelos tabulares guardan candidatos y configuración seleccionada; se verificaron sus predicciones al recargar el modelo. Los controles B tienen exactamente el mismo número de parámetros que B completo; las ramas excluidas reciben ceros constantes y sus sesgos siguen siendo entrenables.', '',
              'Bootstrap: 2000 remuestreos, semilla 20260928. Dentro de cada split se remuestrean pacientes con todas sus imágenes. Para la media de repeticiones se remuestrea la unión de pacientes de test y se aplica el mismo peso a cada aparición del paciente. Se rechazan remuestreos sin alguna de las seis clases y se registran los rechazos.', '',
              'Los intervalos son condicionales a los modelos ya entrenados y a la selección clínica ya realizada. Capturan agrupación por paciente y sus apariciones en test; no capturan la incertidumbre adicional de reentrenar ni la dependencia inducida por train compartido. No equivalen a una evaluación confirmatoria independiente. No se usaron p-valores ni se declaró equivalencia.', '',
              'Calibración: Brier, ECE y tablas de confiabilidad de probabilidades originales, no calibradores ajustados con test. Binario evalúa probabilidad de malignidad; multiclase evalúa confianza top-label. Las matrices de confusión, recalls por clase, especificidad binaria y métricas por lesión están en analysis.json.', '',
              'Queda como ampliación de investigación validar en nuevos datos/pacientes y estudiar incertidumbre de reentrenamiento con un diseño externo o anidado. No se afirma utilidad clínica ni generalización a otros datasets.', '']
    (OUT/'RESULTADOS_TESIS_COMPLETOS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
