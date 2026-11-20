
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from pca_svm_custom_pca import CustomPCA
from pca_svm_har_data import read_shared_splits, verify_participants


class TestPCARBFSupportVectorMachine(unittest.TestCase):
    def test_custom_pca_matches_reference_subspace(self):
        X = np.random.default_rng(71).normal(size=(200, 16))
        mine = CustomPCA(5).fit(X)
        ref = PCA(5, svd_solver='full').fit(X)
        np.testing.assert_allclose(mine.components_.T @ mine.components_,
                                   ref.components_.T @ ref.components_, atol=1e-8)
        np.testing.assert_allclose(mine.explained_variance_ratio_,
                                   ref.explained_variance_ratio_, atol=1e-8)

    def test_bad_pca_input(self):
        with self.assertRaises(ValueError):
            CustomPCA(25).fit(np.ones((20, 6)))
        with self.assertRaises(ValueError):
            CustomPCA(2).fit(np.array([[float('nan'), 1], [1, 2]]))

    def test_split_validation(self):
        train = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
        test = [2, 4, 9, 10, 12, 13, 18, 20, 24]
        folds = [[8, 23, 29], [7, 16, 22], [11, 25, 28], [17, 21, 27],
                 [3, 5, 19], [14, 15, 26], [1, 6, 30]]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'participants.json'
            p.write_text(json.dumps({'seed': 42, 'n_folds': 7, 'train_participants': train,
                                     'test_participants': test, 'cv_folds': folds}))
            self.assertEqual(read_shared_splits(p)['cv_folds'], folds)
            with self.assertRaises(ValueError):
                verify_participants(np.array([1, 2]), train, 'train')
            with self.assertRaises(FileNotFoundError):
                read_shared_splits(p.with_name('not_found.json'))

    def test_end_to_end_seven_folds_without_test_peek(self):
        rng = np.random.default_rng(24)
        train = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
        test = [2, 4, 9, 10, 12, 13, 18, 20, 24]
        folds = [[8, 23, 29], [7, 16, 22], [11, 25, 28], [17, 21, 27],
                 [3, 5, 19], [14, 15, 26], [1, 6, 30]]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            p = base / 'data' / 'splits' / 'participants.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'seed': 42, 'n_folds': 7, 'train_participants': train,
                                     'test_participants': test, 'cv_folds': folds}))
            root = base / 'data' / 'raw' / 'UCI HAR Dataset'
            for label, subjects in [('train', train), ('test', test)]:
                d = root / label
                d.mkdir(parents=True)
                subject_ids = np.repeat(subjects, 12)
                y = np.tile(np.repeat(np.arange(1, 7), 2), len(subjects))
                X = rng.normal(size=(len(y), 18)) + (y[:, None] - 3.0) * .6
                np.savetxt(d / f'X_{label}.txt', X)
                np.savetxt(d / f'y_{label}.txt', y, fmt='%d')
                np.savetxt(d / f'subject_{label}.txt', subject_ids, fmt='%d')
            out = base / 'reports' / 'pca_rbf_svm'
            base_cmd = [sys.executable, str(REPO / 'src' / 'pca_rbf_svm.py'),
                        '--dataset', str(root), '--splits', str(p), '--out', str(out),
                        '--components', '4,8', '--c-values', '1']
            first = subprocess.run(base_cmd, text=True, capture_output=True, timeout=120)
            self.assertEqual(first.returncode, 0, first.stdout + '\n' + first.stderr)
            d = json.loads((out / 'run_summary.json').read_text())
            self.assertFalse(d['official_test_evaluated'])
            self.assertFalse((out / 'test_confusion.csv').exists())
            self.assertEqual(len(d['cv_folds']), 7)
            self.assertEqual(len(d['training_participants']), 21)
            self.assertTrue((out / 'validation_confusion.png').exists())
            self.assertTrue((out / 'validation_metrics.json').exists())
            self.assertEqual(json.loads((out / 'validation_metrics.json').read_text())['train_seconds'] > 0, True)
            second = subprocess.run(base_cmd + ['--final-test'], text=True, capture_output=True, timeout=120)
            self.assertEqual(second.returncode, 0, second.stdout + '\n' + second.stderr)
            d = json.loads((out / 'run_summary.json').read_text())
            self.assertTrue(d['official_test_evaluated'])
            self.assertEqual(len(json.loads((out / 'test_metrics.json').read_text())['per_participant']), 9)
            self.assertTrue((out / 'test_confusion.csv').exists())


if __name__ == '__main__':
    unittest.main()
