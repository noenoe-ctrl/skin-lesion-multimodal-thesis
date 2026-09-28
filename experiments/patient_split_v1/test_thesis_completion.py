import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from complete_thesis_experiments import ZeroMetadata, save_tabular
from protocol import TrainMetadata
from test_metadata_baseline import frame
from thesis_metrics import weighted_primary


class CompletionTests(unittest.TestCase):
    def test_cluster_bootstrap_preserves_patient_weights_across_images_and_repeats(self):
        from analyze_thesis_completion import cluster_weights
        first = frame('first'); second = frame('second')
        first['patient_id'] = [f'p_{i//2}' for i in range(len(first))]
        second['patient_id'] = first.patient_id.to_numpy()
        draws,rejected,patients = cluster_weights([first,second],40,np.random.default_rng(42))
        self.assertEqual(patients,12)
        np.testing.assert_array_equal(draws[0],draws[1])
        np.testing.assert_array_equal(draws[0][:,::2],draws[0][:,1::2])
        for cls in first.diagnostic.unique():
            self.assertTrue(np.all(draws[0][:,first.diagnostic.to_numpy()==cls].sum(axis=1)>0))

    def test_weighted_auc_matches_sklearn_including_ties(self):
        y = np.array([0,1,0,1,0,1]); prob = np.array([.1,.2,.2,.4,.4,.4])
        draws = np.array([[1,1,1,1,1,1],[0,3,2,1,1,2],[2,1,0,4,2,0]])
        actual = weighted_primary(y,prob,'binary',draws)
        expected = [roc_auc_score(y,prob,sample_weight=w) for w in draws]
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_weighted_bacc_matches_sklearn(self):
        y = np.tile(np.arange(6),3); prob = np.eye(6)[np.roll(y,1)]
        weights = np.array([np.ones(len(y)),np.arange(len(y))+1])
        actual = weighted_primary(y,prob,'multi',weights)
        expected = [balanced_accuracy_score(y,prob.argmax(axis=1),sample_weight=w) for w in weights]
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_zero_metadata_removes_information_keeps_dimension(self):
        data = frame('train'); proc = TrainMetadata().fit(data)
        masked = ZeroMetadata(proc)
        np.testing.assert_array_equal(masked.transform(data),np.zeros_like(proc.transform(data)))
        self.assertEqual(masked.state,proc.state)

    def test_tabular_models_use_validation_and_preserve_processor(self):
        parts = {s:frame(s) for s in ['train','val','test']}
        proc = TrainMetadata().fit(parts['train']); state = copy.deepcopy(proc.state)
        with tempfile.TemporaryDirectory() as folder:
            for task in ['binary','multi']:
                for name in ['metadata_logreg','metadata_boosting','metadata_prior']:
                    dest = Path(folder)/task/name
                    met = save_tabular(parts,proc,name,task,42,dest)
                    candidates = json.loads((dest/'candidate_history.json').read_text())
                    config = json.loads((dest/'configuration.json').read_text())
                    self.assertEqual(met['best_validation_score'],max(c['selection_score'] for c in candidates))
                    self.assertFalse(config['refit_train_plus_val'])
                    self.assertEqual(config['preprocessor'],state)
        self.assertEqual(proc.state,state)


if __name__ == '__main__':
    unittest.main()
