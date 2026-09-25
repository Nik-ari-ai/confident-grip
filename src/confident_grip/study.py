"""Подбор на S1 и проверка закреплённого протокола на S2 и S3."""

import argparse
import csv
import hashlib
import json
import platform
import scipy
import sklearn
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from .data import load_db2
from .processing import prepare, fit_quality
from .experiment import SPLIT, predict, evaluate, plot_example, benchmark


def log_features(x):
    """Первые 24 признака — амплитудные; остальные — счётчики."""
    return np.hstack([np.log(np.maximum(x[:, :24], 1e-12)), np.log1p(x[:, 24:])])


def fit_model(train, config):
    x = np.concatenate([p['features'][p['train_ok']] for p in train])
    y = np.concatenate([p['labels'][p['train_ok']] for p in train])
    groups = np.concatenate([np.full(p['train_ok'].sum(), p['trial'].repetition) for p in train])
    classifier = (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                  if config['model'] == 'LDA' else SVC(C=config['C'], gamma='scale'))
    steps = []
    if config['features'] == 'log_td':
        steps.append(FunctionTransformer(log_features))
    base = make_pipeline(*steps, StandardScaler(), classifier)
    folds = list(GroupKFold(3).split(x, y, groups))
    # OOF-оценки калибруются по повторам, итоговый классификатор учится на всех train.
    model = CalibratedClassifierCV(base, method='sigmoid', cv=folds, ensemble=False)
    model.fit(x, y)
    return model


def partition(trials, window_ms):
    # Сначала распределяем целые повторы, затем строим окна внутри каждой части.
    return {name: [prepare(t, window_ms=window_ms) for t in trials if t.repetition in reps]
            for name, reps in SPLIT.items()}


def select_gate(validation):
    candidates = []
    for threshold in (0., .5, .65, .8, .9):
        for margin in (0., .15, .3):
            for confirmations in (1, 2, 3):
                params = dict(threshold=threshold, margin=margin, confirmations=confirmations)
                metrics, _ = evaluate(validation, **params)
                candidates.append({'settings': params, 'metrics': metrics})
    # Не принимаем вариант, который выигрывает за счёт почти полного отказа.
    feasible = [c for c in candidates if c['metrics']['accepted_fraction'] >= .70]
    if not feasible:
        return dict(threshold=0., margin=0., confirmations=1), candidates, False
    best = min(feasible, key=lambda c: (c['metrics']['wrong_fraction_accepted'],
                                       -c['metrics']['response_within_100ms_fraction'],
                                       -c['metrics']['accepted_fraction']))
    return best['settings'], candidates, True


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def tune(data, output):
    trials = [t for t in load_db2(data) if t.repetition in (1, 2, 3, 4)]
    candidates = []
    for window in (50, 75, 100, 150):
        parts = partition(trials, window)
        quality = fit_quality(parts['train'])
        for feature_set in ('td', 'log_td'):
            for name, c in (('LDA', None), ('SVM', 1), ('SVM', 10)):
                config = dict(model=name, C=c, features=feature_set, window_ms=window)
                model = fit_model(parts['train'], config)
                prediction = predict(model, parts['validation'], quality)
                metrics, _ = evaluate(prediction)
                candidates.append({'config': config, 'metrics': metrics})
                print(config, 'F1', round(metrics['raw_macro_f1'], 4), flush=True)
                save(output/'search.json', candidates)
    chosen = {}
    for name in ('LDA', 'SVM'):
        best = max([c for c in candidates if c['config']['model'] == name],
                   key=lambda c: (c['metrics']['raw_macro_f1'], -c['config']['window_ms']))
        chosen[name] = best['config']
    save(output/'frozen_config.json', {'development_subject': 'S1', 'split': SPLIT,
        'selection': 'max validation macro F1, tie: shorter window',
        'gate_selection': 'minimum accepted error at validation coverage >= 0.70',
        'gate_targets': {'max_accepted_error': .10, 'min_coverage': .70,
                         'min_response_within_100ms': .90},
        'models': chosen, 'new_subjects': ['S2', 'S3'],
        'note': 'На новом участнике веса и пороги настраиваются персонально по train/validation. Это не перенос весов без калибровки.'})


def test(data, output, config_path):
    frozen = json.loads(config_path.read_text())
    trials = load_db2(data)
    report = {'environment': {'python': platform.python_version(), 'system': platform.system(),
                              'machine': platform.machine(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'file': data.name, 'sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'split': SPLIT, 'models': {}}
    output.mkdir(parents=True, exist_ok=True)
    for name, config in frozen['models'].items():
        parts = partition(trials, config['window_ms'])
        model = fit_model(parts['train'], config)
        quality = fit_quality(parts['train'])
        validation = predict(model, parts['validation'], quality)
        settings, candidates, feasible = select_gate(validation)
        testing = predict(model, parts['test'], quality)
        variants = {'baseline': dict(threshold=0., margin=0., confirmations=1),
                    'threshold_only': dict(threshold=settings['threshold'], margin=settings['margin'], confirmations=1),
                    'confirmation_only': dict(threshold=0., margin=0., confirmations=settings['confirmations']),
                    'combined': settings}
        item = {'config': config, 'settings': settings, 'coverage_feasible_on_validation': feasible,
                'validation_candidates': candidates, 'test': {}}
        for variant, params in variants.items():
            metrics, rows = evaluate(testing, **params)
            targets = frozen['gate_targets']
            metrics['research_targets_met'] = bool(
                metrics['wrong_fraction_accepted'] is not None and
                metrics['wrong_fraction_accepted'] <= targets['max_accepted_error'] and
                metrics['accepted_fraction'] >= targets['min_coverage'] and
                metrics['response_within_100ms_fraction'] >= targets['min_response_within_100ms'])
            item['test'][variant] = metrics
            with (output/f'{name.lower()}_{variant}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(['trial', 'time_ms', 'label', 'prediction', 'command', 'inhibit', 'reason'])
                writer.writerows(rows)
            if variant == 'combined':
                plot_example(rows, output/f'{name.lower()}_trace.png')
        truth = np.concatenate([p['labels'] for p in testing])
        raw = np.concatenate([p['scores'].argmax(axis=1) for p in testing])
        item['confusion_matrix'] = confusion_matrix(truth, raw, labels=range(6)).tolist()
        item['per_class'] = classification_report(truth, raw, labels=range(6), output_dict=True, zero_division=0)
        item['timing'] = benchmark(model, parts['test'][0]['trial'], settings, quality, config['window_ms'])
        report['models'][name] = item
        print(data.name, name, settings, item['test']['combined'], flush=True)
        save(output/'metrics.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['tune', 'test'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.mode == 'tune':
        tune(args.data, args.output)
    elif args.config is None:
        parser.error('Для test нужен --config')
    else:
        test(args.data, args.output, args.config)


if __name__ == '__main__':
    main()
