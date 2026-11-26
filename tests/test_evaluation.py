from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from export_numpy_bundle import fit_evaluated_classifier, export_clf
import grid_lib as gl
import grid_mlp as mlp


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(120, 2)), columns=['a', 'b'])
        self.y = pd.Series((self.X['a'] > 0).astype(int))
        self.tr, self.va, self.te = np.arange(80), np.arange(80, 100), np.arange(100, 120)

    def fit(self, y=None, X=None, va=None):
        return fit_evaluated_classifier(self.X if X is None else X, self.y if y is None else y,
                                       self.tr, self.va if va is None else va, self.te,
                                       MLPClassifier(hidden_layer_sizes=(4,), solver='lbfgs', max_iter=500, random_state=4))

    def test_test_labels_cannot_change_fit_or_threshold(self):
        c1, s1, t1, e1 = self.fit()
        changed = self.y.copy()
        changed.loc[self.te] = 1 - changed.loc[self.te]
        c2, s2, t2, e2 = self.fit(y=changed)
        self.assertEqual(t1, t2)
        for a, b in zip(c1.coefs_, c2.coefs_):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(s1.mean_, self.X.loc[self.tr].mean())
        self.assertNotEqual(e1['test']['confusion'], e2['test']['confusion'])
        self.assertEqual(e1['threshold_source'], 'validation')
        np.testing.assert_allclose(mlp.np_forward_prob(export_clf(c1, s1), self.X),
                                   c1.predict_proba(s1.transform(self.X))[:, 1], atol=1e-14)

    def test_validation_and_test_values_not_in_scaler(self):
        X = self.X.copy()
        X.loc[80:] += 1000
        _, sc, _, _ = self.fit(X=X)
        np.testing.assert_allclose(sc.mean_, self.X.loc[self.tr].mean())

    def test_bad_splits_rejected(self):
        for va in [[], [0, 80], [80, 80]]:
            with self.subTest(va=va), self.assertRaises(ValueError):
                self.fit(va=va)

    def test_single_class_validation_and_test(self):
        y = self.y.copy()
        y.loc[80:] = 0
        _, _, threshold, ev = self.fit(y=y)
        self.assertEqual(threshold, .5)
        self.assertIsNone(ev['test']['AUC'])
        self.assertIsNone(ev['test']['F1_macro'])

    def test_metrics_empty_and_single_class(self):
        with self.assertRaises(ValueError):
            gl.clf_metrics([], [])
        metrics = gl.clf_metrics([0, 0], [.1, .9])
        self.assertEqual(metrics['confusion'], [[1, 1], [0, 0]])
        self.assertIsNone(metrics['AUC'])


if __name__ == '__main__':
    unittest.main()
