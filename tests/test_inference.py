"""Регрессии инференса на поставляемых весах; без обучения и перезаписи артефактов."""
from pathlib import Path
import copy
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import baseline_solution as entry
import grid_mlp as mlp
from grid_validation import validate_predictions
import grid_lib as gl


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = pd.read_csv(ROOT / 'my_test/nodes.csv')
        cls.lines = pd.read_csv(ROOT / 'my_test/lines.csv')
        cls.bundle = joblib.load(ROOT / 'models/weights_v2_numpy.joblib')
        cls.pn, cls.pl, _, _ = mlp.predict_bundle(cls.nodes, cls.lines, cls.bundle)

    def predict(self, nodes=None, lines=None, bundle=None):
        return mlp.predict_bundle(self.nodes if nodes is None else nodes,
                                  self.lines if lines is None else lines,
                                  self.bundle if bundle is None else bundle)

    def test_matches_before_changes(self):
        # Исторический прогон был сохранён до правок. Если его нет, сравним
        # с поставляемой выдачей: это проверка неизменности, не качества.
        before = ROOT / 'predictions_check/before'
        if not before.is_dir():
            before = ROOT / 'predictions'
        for name, actual in [('nodes', self.pn), ('lines', self.pl)]:
            expected = pd.read_csv(before / f'pred_{name}.csv')
            pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-11, atol=1e-10)

    def test_targets_unused(self):
        n = self.nodes.drop(columns=['V', 'Delta', 'regime_valid'])
        l = self.lines.drop(columns=['I_max', 'I_zarg', 'I_доп'])
        a, b, _, _ = self.predict(n, l)
        pd.testing.assert_frame_equal(a, self.pn)
        pd.testing.assert_frame_equal(b, self.pl)
        for col in ['V', 'Delta', 'regime_valid']:
            n[col] = 'ignored target'
        a, _, _, _ = self.predict(n, l)
        pd.testing.assert_frame_equal(a, self.pn)

    def test_input_order_preserved(self):
        n = self.nodes.sample(frac=1, random_state=3)
        l = self.lines.sample(frac=1, random_state=4)
        a, b, _, _ = self.predict(n, l)
        for actual, expected in [(a, self.pn), (b, self.pl)]:
            pd.testing.assert_frame_equal(actual.sort_values(['ID_опыта', 'Номер']).reset_index(drop=True),
                                          expected.sort_values(['ID_опыта', 'Номер']).reset_index(drop=True))
        entry.validate(n, l, a, b)

    def test_missing_slack_allowed(self):
        a, _, _, _ = self.predict(self.nodes[self.nodes['Номер'] != 999])
        pd.testing.assert_frame_equal(a, self.pn[self.pn['Номер'] != 999].reset_index(drop=True))

    def test_invalid_inputs_rejected(self):
        variants = []
        variants.append((pd.concat([self.nodes, self.nodes.iloc[:1]]), self.lines))
        variants.append((self.nodes.drop(index=self.nodes.index[0]), self.lines))
        variants.append((self.nodes, self.lines.iloc[:-1]))
        variants.append((self.nodes.iloc[:0], self.lines))
        variants.append((self.nodes.drop(columns=['P_н']), self.lines))
        for col, value in [('P_н', np.nan), ('P_г', np.inf), ('Номер', 1.5), ('U_ном', 0)]:
            n = self.nodes.astype({col: float}).copy()
            n.loc[n.index[0], col] = value
            variants.append((n, self.lines))
        l = self.lines.copy()
        l.loc[l.index[-1], 'ID_опыта'] = 987654
        variants.append((self.nodes, l))
        l = self.lines.copy()
        l.loc[l.index[-1], 'R'] += 1
        variants.append((self.nodes, l))
        n = self.nodes.copy()
        n['P н'] = n['P_н']
        variants.append((entry.normalize(n, entry.NODE_ALIASES), self.lines))
        for i, (n, l) in enumerate(variants):
            with self.subTest(case=i), self.assertRaises(ValueError):
                self.predict(n, l)

    def test_bad_bundle_rejected(self):
        b = copy.deepcopy(self.bundle)
        b['reg']['v']['x_scale'][0] = 0
        with self.assertRaises(ValueError):
            self.predict(bundle=b)

    def test_zero_invalid_and_output_validation(self):
        b = copy.deepcopy(self.bundle)
        # Гарантированно отрицательный логит, без предположений о насыщении весов.
        W, bias = b['clf']['coefs'][-1]
        b['clf']['coefs'][-1] = (np.zeros_like(W), np.full_like(bias, -100.))
        a, c, _, _ = self.predict(bundle=b)
        self.assertTrue((a['Regime_valid'] == 0).all())
        self.assertTrue((a[['V', 'Delta']] == 0).all().all())
        self.assertTrue((c[['Imax, A', 'Izarg, %']] == 0).all().all())
        entry.validate(self.nodes, self.lines, a, c)
        a.loc[0, 'V'] = 1
        with self.assertRaises(ValueError):
            entry.validate(self.nodes, self.lines, a, c)

    def test_parallel_and_current_conversion(self):
        inv = {}
        for line, group in gl.parallel_groups(self.lines).items():
            inv.setdefault(group, []).append(line)
        wide = self.pl.pivot(index='ID_опыта', columns='Номер', values='Izarg, %')
        for members in inv.values():
            if len(members) > 1:
                self.assertLessEqual(float(wide[members].max(axis=1).sub(wide[members].min(axis=1)).max()), 1e-12)
        nom = self.pl['Номер'].map(self.bundle['idop_nominal'])
        np.testing.assert_allclose(self.pl['Imax, A'], self.pl['Izarg, %'] * nom / 100)

    def test_sigmoid_extreme_values(self):
        md = {'x_mean': np.array([0.]), 'x_scale': np.array([1.]),
              'coefs': [(np.ones((1, 1)), np.zeros(1))]}
        with np.errstate(over='raise'):
            np.testing.assert_array_equal(mlp.np_forward_prob(md, [[-1000.], [1000.]]), [0., 1.])

    def test_cli_success_and_missing_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'ok'
            args = [sys.executable, str(ROOT / 'baseline_solution.py'), '--nodes', str(ROOT / 'my_test/nodes.csv'),
                    '--lines', str(ROOT / 'my_test/lines.csv'), '--out-dir', str(out)]
            result = subprocess.run(args, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / 'pred_nodes.csv').is_file())
            corrupt = Path(tmp) / 'bad.joblib'
            corrupt.write_bytes(b'not a model')
            for weights in [Path(tmp) / 'absent.joblib', corrupt]:
                bad_out = Path(tmp) / weights.stem
                bad_args = args.copy()
                bad_args[bad_args.index('--out-dir') + 1] = str(bad_out)
                result = subprocess.run(bad_args + ['--weights', str(weights)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((bad_out / 'pred_nodes.csv').exists())

    def test_numpy_inference_without_sklearn_imports(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tests/check_numpy_only.py')], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
