"""Отдельный процесс: запрет sklearn до импорта приложения и загрузки весов."""
import importlib.abc
from pathlib import Path
import sys
import tempfile


class NoSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise ImportError('sklearn intentionally blocked')
        return None


sys.meta_path.insert(0, NoSklearn())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import baseline_solution

with tempfile.TemporaryDirectory() as tmp:
    sys.argv = ['baseline_solution.py', '--nodes', str(ROOT / 'my_test/nodes.csv'),
                '--lines', str(ROOT / 'my_test/lines.csv'), '--out-dir', tmp]
    baseline_solution.main()
    if not (Path(tmp) / 'pred_nodes.csv').is_file() or any(k.startswith('sklearn') for k in sys.modules):
        raise RuntimeError('numpy-only check failed')
print('numpy-only inference passed')
