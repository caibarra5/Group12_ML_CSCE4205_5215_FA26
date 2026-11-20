
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pca_svm_custom_pca import CustomPCA
from pca_svm_har_data import ACTIVITY_NAMES, LABELS, load_har, read_shared_splits, verify_participants
from metrics import evaluate, timed  # exact shared metric API from src/metrics.py


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list, columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_positive_unique(value: str, cast):
    try:
        vals = [cast(s.strip()) for s in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError('Expected comma-separated numeric values') from e
    if not vals or len(vals) != len(set(vals)) or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError('Values must be positive and unique')
    return vals


def pipeline_for(k: int, c: float):
    return Pipeline([
        ('scale', StandardScaler()),
        ('pca', CustomPCA(n_components=k)),
        ('svm', SVC(kernel='rbf', gamma='scale', C=c, cache_size=512)),
    ])


def plot_confusion(confusion, out: Path, title: str):
    cm = np.asarray(confusion, dtype=int)
    names = list(ACTIVITY_NAMES.values())
    rows = [{'actual': name, **{names[j]: int(cm[i, j]) for j in range(6)}}
            for i, name in enumerate(names)]
    write_csv(out.with_suffix('.csv'), rows, ['actual', *names])
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(6), names, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(6), names, fontsize=8)
    ax.set_xlabel('Predicted activity')
    ax.set_ylabel('True activity')
    ax.set_title(title)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > .55 * max(1, int(cm.max())) else 'black')
    fig.colorbar(im, ax=ax)
    fig.savefig(out.with_suffix('.png'), dpi=160)
    plt.close(fig)


def save_participant_rows(shared_metrics: dict, subjects, out: Path):
    counts = {int(pid): int(np.sum(np.asarray(subjects) == int(pid)))
              for pid in np.unique(subjects)}
    rows = [{'participant': int(pid), 'n_windows': counts[int(pid)], **score}
            for pid, score in shared_metrics['per_participant'].items()]
    write_csv(out, rows, ['participant', 'n_windows', 'macro_f1', 'accuracy'])
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.bar([str(r['participant']) for r in rows], [r['macro_f1'] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Participant ID')
    ax.set_ylabel('Macro-F1 (shared metrics definition)')
    ax.set_title('PCA + RBF-SVM: per-participant macro-F1')
    fig.savefig(out.with_suffix('.png'), dpi=160)
    plt.close(fig)


def extra_macro_metrics(y_true, y_pred):
    return {
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0)),
    }


def run(args):
    split = read_shared_splits(args.splits)
    train = load_har(args.dataset, 'train')
    verify_participants(train.subjects, split['train_participants'], 'train')
    # Reading participant IDs is safe for a split integrity audit; read no test X or y.
    test_subject_file = args.dataset / 'test' / 'subject_test.txt'
    if test_subject_file.exists():
        expected_test_subjects = np.loadtxt(test_subject_file, dtype=np.int64, ndmin=1)
        verify_participants(expected_test_subjects, split['test_participants'], 'test')

    ks = parse_positive_unique(args.components, int)
    cs = parse_positive_unique(args.c_values, float)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if max(ks) > min(len(train.X), train.X.shape[1]):
        raise ValueError('Requested more PCA components than training rows/features')

    configurations = [(k, c) for k in ks for c in cs]
    predictions = {key: np.full(len(train.y), -1, dtype=np.int64) for key in configurations}
    fold_rows = []
    fold_times = {key: [0.0, 0.0] for key in configurations}
    for fold_idx, ids in enumerate(split['cv_folds'], 1):
        val = np.isin(train.subjects, ids)
        fit = ~val
        if not val.any() or not fit.any():
            raise ValueError('Empty training/validation fold')
        if set(train.subjects[fit]) & set(train.subjects[val]):
            raise AssertionError('Participant data leakage')
        if set(train.subjects[val]) != set(ids):
            raise ValueError('Fold subjects differ from participants.json')
        if max(ks) > min(fit.sum(), train.X.shape[1]):
            raise ValueError('A candidate PCA dimension exceeds this fold size')

        # Compute the SAME training-only PCA once per fold; the smaller candidates
        # take the leading eigenvectors of the fitted max-k basis.
        start = time.perf_counter()
        scaler = StandardScaler().fit(train.X[fit])
        z_fit = scaler.transform(train.X[fit])
        z_val = scaler.transform(train.X[val])
        pca = CustomPCA(n_components=max(ks)).fit(z_fit)
        projected_fit = pca.transform(z_fit)
        prep_seconds = time.perf_counter() - start

        for k, c in configurations:
            svm = SVC(kernel='rbf', gamma='scale', C=c, cache_size=512)
            _, svm_fit_seconds = timed(svm.fit, projected_fit[:, :k], train.y[fit])
            # Time the *whole* inference path, including scaler + PCA + SVM.
            predicted, svm_pred_seconds = timed(
                lambda: svm.predict(pca.transform(scaler.transform(train.X[val]))[:, :k])
            )
            predictions[(k, c)][val] = predicted
            # REQUIRED: use the shared evaluator for exactly the same metrics
            # and data slicing as David and Christopher.
            scored = evaluate(train.y[val], predicted, train.subjects[val],
                              train_seconds=float(prep_seconds + svm_fit_seconds),
                              predict_seconds=float(svm_pred_seconds), labels=LABELS)
            fold_times[(k, c)][0] += float(prep_seconds + svm_fit_seconds)
            fold_times[(k, c)][1] += float(svm_pred_seconds)
            fold_rows.append({
                'fold': fold_idx, 'validation_subjects': '-'.join(map(str, ids)),
                'n_train_windows': int(fit.sum()), 'n_val_windows': int(val.sum()),
                'pca_components': k, 'C': c, 'macro_f1': scored['macro_f1'],
                'accuracy': scored['accuracy'],
                **extra_macro_metrics(train.y[val], predicted),
                'train_seconds': scored['train_seconds'],
                'predict_seconds': scored['predict_seconds'],
            })
        print(f'Finished fold {fold_idx}/{len(split["cv_folds"])}: participants {ids}', flush=True)

    for key, pred in predictions.items():
        if np.any(pred == -1):
            raise AssertionError(f'Missing out-of-fold predictions for {key}')

    write_csv(out / 'cv_fold_metrics.csv', fold_rows, [
        'fold', 'validation_subjects', 'n_train_windows', 'n_val_windows',
        'pca_components', 'C', 'macro_f1', 'accuracy', 'macro_precision',
        'macro_recall', 'train_seconds', 'predict_seconds',
    ])
    summary_rows = []
    for k, c in configurations:
        rows = [r for r in fold_rows if r['pca_components'] == k and r['C'] == c]
        scores = np.asarray([r['macro_f1'] for r in rows])
        combined = evaluate(train.y, predictions[(k, c)], train.subjects,
                            train_seconds=fold_times[(k, c)][0],
                            predict_seconds=fold_times[(k, c)][1], labels=LABELS)
        summary_rows.append({
            'pca_components': k, 'C': c,
            'mean_fold_macro_f1': float(scores.mean()),
            'sd_fold_macro_f1': float(scores.std(ddof=0)),
            'oof_macro_f1': combined['macro_f1'],
            'oof_accuracy': combined['accuracy'],
            'cv_total_train_seconds': fold_times[(k, c)][0],
            'cv_total_predict_seconds': fold_times[(k, c)][1],
        })
    write_csv(out / 'cv_grid_summary.csv', summary_rows, [
        'pca_components', 'C', 'mean_fold_macro_f1', 'sd_fold_macro_f1',
        'oof_macro_f1', 'oof_accuracy', 'cv_total_train_seconds', 'cv_total_predict_seconds',
    ])
    # Fixed before touching the held-out test split; smaller k, then C wins ties.
    best = min(summary_rows, key=lambda r: (-r['mean_fold_macro_f1'], r['pca_components'], r['C']))
    best_k, best_c = int(best['pca_components']), float(best['C'])
    chosen_pred = predictions[(best_k, best_c)]
    oof = evaluate(train.y, chosen_pred, train.subjects,
                   train_seconds=fold_times[(best_k, best_c)][0],
                   predict_seconds=fold_times[(best_k, best_c)][1], labels=LABELS)
    oof.update(extra_macro_metrics(train.y, chosen_pred))
    write_json(out / 'validation_metrics.json', oof)
    plot_confusion(oof['confusion_matrix'], out / 'validation_confusion',
                   'PCA + RBF-SVM: out-of-fold validation')
    save_participant_rows(oof, train.subjects, out / 'validation_per_participant.csv')
    write_csv(out / 'validation_predictions.csv', [
        {'participant': int(s), 'actual': int(y), 'predicted': int(p)}
        for s, y, p in zip(train.subjects, train.y, chosen_pred)
    ], ['participant', 'actual', 'predicted'])

    # Fit final model only on the 21 official training participants.
    selected = pipeline_for(best_k, best_c)
    _, final_fit_seconds = timed(selected.fit, train.X, train.y)
    pca_fitted = selected.named_steps['pca']
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(np.arange(1, best_k + 1), np.cumsum(pca_fitted.explained_variance_ratio_))
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Retained PCA components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('PCA variance learned on official training participants')
    fig.savefig(out / 'pca_training_explained_variance.png', dpi=160)
    plt.close(fig)

    manifest = {
        'method': 'Student-written NumPy PCA + sklearn RBF-SVM',
        'component': 'PCA + RBF-SVM',
        'dataset_representation': '561 engineered UCI HAR features',
        'shared_split': args.splits.as_posix(),
        'seed': split['seed'],
        'training_participants': split['train_participants'],
        'test_participants': split['test_participants'],
        'cv_folds': split['cv_folds'],
        'selection_metric': 'highest unweighted mean across seven held-out participant fold macro-F1',
        'tie_break': 'smaller PCA dimension, then smaller C',
        'candidate_pca_components': ks,
        'candidate_C': cs,
        'svm_gamma': 'scale',
        'selected_pca_components': best_k,
        'selected_C': best_c,
        'mean_validation_macro_f1': best['mean_fold_macro_f1'],
        'validation_sd_macro_f1': best['sd_fold_macro_f1'],
        'out_of_fold_validation_macro_f1': oof['macro_f1'],
        'out_of_fold_validation_accuracy': oof['accuracy'],
        'final_fit_seconds': final_fit_seconds,
        'final_training_explained_variance': float(np.sum(pca_fitted.explained_variance_ratio_)),
        'official_test_evaluated': False,
        'environment': {
            'python': sys.version.split()[0],
            'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
            'os': platform.platform(), 'cpu': platform.processor() or 'not reported',
            'logical_cpus': os.cpu_count(),
        },
    }
    if args.final_test:
        # THIS FLAG MUST ONLY BE USED AFTER THE THREE METHODS ARE LOCKED.
        test = load_har(args.dataset, 'test')
        verify_participants(test.subjects, split['test_participants'], 'test')
        test_pred, test_predict_seconds = timed(selected.predict, test.X)
        test_metrics = evaluate(test.y, test_pred, test.subjects,
                                train_seconds=final_fit_seconds,
                                predict_seconds=float(test_predict_seconds), labels=LABELS)
        test_metrics.update(extra_macro_metrics(test.y, test_pred))
        test_metrics['n_test_windows'] = int(len(test.y))
        test_metrics['prediction_ms_per_window'] = 1000 * test_predict_seconds / len(test.X)
        write_json(out / 'test_metrics.json', test_metrics)
        plot_confusion(test_metrics['confusion_matrix'], out / 'test_confusion',
                       'PCA + RBF-SVM: official test participants')
        save_participant_rows(test_metrics, test.subjects, out / 'test_per_participant.csv')
        manifest['official_test_evaluated'] = True
        manifest['official_test_macro_f1'] = test_metrics['macro_f1']
        manifest['official_test_accuracy'] = test_metrics['accuracy']

    write_json(out / 'run_summary.json', manifest)
    print(f'Selected PCA components={best_k}, C={best_c}', flush=True)
    print(f'Mean fold validation macro-F1={best["mean_fold_macro_f1"]:.5f}', flush=True)
    print(f'Out-of-fold validation macro-F1={oof["macro_f1"]:.5f}', flush=True)
    print(f'Official test evaluated={manifest["official_test_evaluated"]}', flush=True)
    print(f'Results written to {out}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/raw/UCI HAR Dataset'))
    parser.add_argument('--splits', type=Path, default=Path('data/splits/participants.json'))
    parser.add_argument('--out', type=Path, default=Path('experiments/pca_rbf_svm'))
    parser.add_argument('--components', default='40,80,160', help='PCA counts to validate')
    parser.add_argument('--c-values', default='0.1,1,10', help='RBF-SVM C values to validate')
    parser.add_argument('--final-test', action='store_true',
                        help='Use only when all method/hyperparameter choices are frozen')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
