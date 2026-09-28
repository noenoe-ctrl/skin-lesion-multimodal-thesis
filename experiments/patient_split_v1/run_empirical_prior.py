"""Unweighted empirical train-frequency reference, additional to weighted prior."""
import json
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from protocol import DEFAULT_OUTPUT, SEEDS, TrainMetadata, digest, write_json
from thesis_metrics import target, evaluate_probabilities, prediction_frame


def main():
    dest = DEFAULT_OUTPUT / 'runs' / 'metadata_empirical_prior_v1'
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in SEEDS:
        split = DEFAULT_OUTPUT / 'splits' / f'seed_{seed}'
        parts = {s:pd.read_csv(split / f'{s}_raw.csv') for s in ['train','val','test']}
        state = json.loads((split / 'preprocessor.json').read_text())
        proc = TrainMetadata(state)
        for task in ['binary','multi']:
            folder = dest / f'seed_{seed}' / task / 'metadata_empirical_prior'
            folder.mkdir(parents=True)
            model = DummyClassifier(strategy='prior').fit(proc.transform(parts['train']),target(parts['train'],task))
            joblib.dump(model,folder / 'model.joblib')
            def probabilities(s):
                p = model.predict_proba(proc.transform(parts[s]))
                return p[:,1] if task == 'binary' else p
            val = evaluate_probabilities(target(parts['val'],task),probabilities('val'),task)
            met = evaluate_probabilities(target(parts['test'],task),probabilities('test'),task)
            met.update(seed=seed,task=task,model='metadata_empirical_prior',loss='prior',
                       best_validation_score=val['auc' if task=='binary' else 'balanced_accuracy'],
                       evaluated_split='test',smoke=False)
            cfg = {'model':'metadata_empirical_prior','task':task,'seed':seed,'preprocessor':state,
                   'train_img_ids':parts['train'].img_id.tolist(),'val_img_ids':parts['val'].img_id.tolist(),
                   'multiclass_train_balanced_weights':False,'refit_train_plus_val':False,
                   'best_validation_score':met['best_validation_score'],'class_prior':model.class_prior_.tolist()}
            write_json(folder / 'configuration.json',cfg); write_json(folder / 'metrics.json',met)
            for s in ['val','test']:
                prediction_frame(parts[s],probabilities(s),task).to_csv(folder / f'{s}_predictions.csv',index=False)
            rows.append(met)
    pd.DataFrame(rows).to_csv(dest / 'metrics_per_seed.csv',index=False)
    write_json(dest / 'run_manifest.json',{'status':'complete','completed_experiments':10,
               'seeds':SEEDS,'class_prior':'unweighted train frequencies',
               'protocol_sha256':digest(DEFAULT_OUTPUT/'protocol.json'), 'code_sha256':digest(__file__)})
    print('COMPLETE: 10 empirical-prior references, no test-based selection.')


if __name__ == '__main__':
    main()
