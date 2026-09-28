"""Small regression checks for leakage, numeric scaling and cache alignment."""
import copy
import unittest

import numpy as np
import pandas as pd

from protocol import CATEGORICAL, META, NUMERIC, TrainMetadata, UNKNOWN


def sample(values):
    frame = pd.DataFrame({c: ['FALSE'] * len(values) for c in CATEGORICAL})
    for c in NUMERIC:
        frame[c] = values
    frame['diagnostic'] = ['BCC'] * len(values)
    return frame


class MetadataTests(unittest.TestCase):
    def test_train_median_applied_to_unseen_data_without_fit(self):
        train = sample([1., 3., np.nan])
        val = sample([1000., np.nan])
        processor = TrainMetadata().fit(train)
        before = copy.deepcopy(processor.state)
        for c in NUMERIC:
            self.assertEqual(processor.state['numeric'][c]['median'], 2.)
            self.assertEqual(processor.impute(val).loc[1, c], 2.)
        processor.transform(val)
        self.assertEqual(before, processor.state)

    def test_numeric_columns_scaled_by_name_and_no_clipping(self):
        processor = TrainMetadata().fit(sample([10., 20.]))
        features = processor.transform(sample([15., 30.]))
        np.testing.assert_allclose(features[:, :4], [[.5] * 4, [2.] * 4])
        self.assertEqual(processor.state['feature_names'][:4], NUMERIC)

    def test_diagnostic_is_never_an_imputation_key(self):
        a = sample([1., 9., np.nan])
        b = a.copy(); b['diagnostic'] = ['SEK', 'MEL', 'NEV']
        self.assertEqual(TrainMetadata().fit(a).state, TrainMetadata().fit(b).state)

    def test_unknown_and_missing_categorical_policy(self):
        train = sample([1., 2., 3.]); train['gender'] = ['MALE', 'MALE', 'FEMALE']
        processor = TrainMetadata().fit(train)
        val = sample([np.nan, 999.]); val['gender'] = ['UNK', 'NEW_CATEGORY']
        self.assertEqual(processor.impute(val).loc[0, 'gender'], 'MALE')
        matrix = processor.transform(val)
        idx = processor.state['feature_names'].index(f'gender={UNKNOWN}')
        self.assertEqual(matrix[1, idx], 1.)
        self.assertNotIn('NEW_CATEGORY', processor.state['categorical']['gender']['categories'])

    def test_empty_numeric_training_column_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'entirely missing'):
            TrainMetadata().fit(sample([np.nan, np.nan]))


if __name__ == '__main__':
    unittest.main()
