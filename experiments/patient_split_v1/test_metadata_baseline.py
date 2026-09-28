"""Integration checks for a strictly metadata-only head and validation selection."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import CLASSES, CATEGORICAL, NUMERIC, TrainMetadata
from run_metadata_baseline import train_metadata


def frame(prefix):
    n = 24
    df = pd.DataFrame({c: ['FALSE', 'TRUE'] * 12 for c in CATEGORICAL})
    for c in NUMERIC:
        df[c] = np.arange(n, dtype=float)
    df.loc[0, 'diameter_1'] = np.nan
    df['diagnostic'] = CLASSES * 4
    df['label'] = df.diagnostic.isin(['BCC', 'MEL', 'SCC']).astype(int)
    df['img_id'] = [f'{prefix}_{i}.png' for i in range(n)]
    df['patient_id'] = [f'{prefix}_{i}' for i in range(n)]
    df['lesion_id'] = 1
    return df


class MetadataHeadTests(unittest.TestCase):
    def test_both_tasks_train_without_visual_features_or_test_and_select_validation(self):
        import torch
        torch.set_num_threads(2)
        parts = {'train': frame('train'), 'val': frame('val')}
        processor = TrainMetadata().fit(parts['train'])
        original = copy.deepcopy(processor.state)
        with tempfile.TemporaryDirectory() as tmp:
            for task, loss in [('binary', 'bce'), ('multi', 'focal')]:
                dest = Path(tmp) / task
                result = train_metadata(parts, processor, task, loss, 42, 2, 2, 'cpu', dest, True)
                self.assertEqual(result['evaluated_split'], 'val')
                self.assertTrue(result['smoke'])
                self.assertFalse((dest / 'test_predictions.csv').exists())
                history = pd.read_csv(dest / 'history.csv')
                self.assertEqual(result['best_epoch'], history.loc[history.selection_score.idxmax(), 'epoch'])
                config = json.loads((dest / 'configuration.json').read_text())
                self.assertEqual(config['preprocessor'], original)
                checkpoint = torch.load(dest / 'head.pt', weights_only=True)
                self.assertEqual(checkpoint['net.0.weight'].shape[1], len(original['feature_names']))
                self.assertTrue(all(np.isfinite(result[k]) for k in ['auc', 'accuracy', 'f1']))
        self.assertEqual(processor.state, original)


if __name__ == '__main__':
    unittest.main()
