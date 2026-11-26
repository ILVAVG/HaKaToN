from pathlib import Path
import copy
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from exp_stage4 import evaluate_test_regression


class SubmissionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.read_csv(ROOT / 'my_test/nodes.csv')
        self.lines = pd.read_csv(ROOT / 'my_test/lines.csv')
        self.bundle = joblib.load(ROOT / 'models/weights_v2_numpy.joblib')
        self.ids = self.nodes['ID_опыта'].unique()

    def test_false_rejection_increases_submitted_error(self):
        b = copy.deepcopy(self.bundle)
        W, bias = b['clf']['coefs'][-1]
        b['clf']['coefs'][-1] = (np.zeros_like(W), np.full_like(bias, -100.))
        conditional, submitted, n_valid = evaluate_test_regression(self.nodes, self.lines, b, self.ids)
        self.assertGreater(n_valid, 0)
        self.assertLess(conditional['V_all_nodes']['MAE'], 1.)
        self.assertGreater(submitted['V_all_nodes']['MAE'], 100.)
        self.assertGreater(submitted['Imax_nominal']['MAE'], conditional['Imax_nominal']['MAE'])

    def test_no_valid_is_undefined(self):
        nodes = self.nodes.copy()
        nodes['regime_valid'] = 0
        conditional, submitted, n_valid = evaluate_test_regression(nodes, self.lines, self.bundle, self.ids)
        self.assertEqual(n_valid, 0)
        self.assertIsNone(conditional['V_all_nodes']['MAE'])
        self.assertIsNone(submitted['Izarg']['MAE'])

    def test_one_valid_experience(self):
        conditional, submitted, count = evaluate_test_regression(self.nodes, self.lines, self.bundle, self.ids[:1])
        self.assertEqual(count, 1)
        self.assertIsNone(conditional['V_uniform_R2_nodes'])
        self.assertIsNone(submitted['V_varying_nodes']['MAE'])
        self.assertEqual(conditional['V_MAE_by_load_quintile'], [])

    def test_constant_load_experiences(self):
        nodes = self.nodes.copy()
        nodes['P_н'] = 0.
        conditional, _, _ = evaluate_test_regression(nodes, self.lines, self.bundle, self.ids)
        self.assertEqual(conditional['V_MAE_by_load_quintile'], [])
        self.assertIn('constant load', conditional['load_quintile_note'])

    def test_summary_missing_values(self):
        from exp_stage4 import summarize_values
        import json
        for values in [[None, None], [None, 1., 3.], []]:
            summary = summarize_values(values)
            json.dumps(summary, allow_nan=False)
            self.assertEqual(summary['n_total'], len(values))
        summary = summarize_values([None, 1., 3.])
        self.assertEqual(summary['mean'], 2.)
        self.assertEqual(summary['n_defined'], 2)
        self.assertIsNone(summarize_values([None])['mean'])
if __name__ == '__main__':
    unittest.main()
