"""Сводка сохранённых опытов без повторного обучения и настройки."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    with path.open() as stream:
        return {(r['trial'], float(r['time_ms'])): r for r in csv.DictReader(stream)}


def main():
    paired = {}
    for subject in ('s2', 's3'):
        old = rows(ROOT/f'results/study/original_{subject}/svm_baseline.csv')
        new = rows(ROOT/f'results/study/{subject}/svm_baseline.csv')
        keys = old.keys() & new.keys()
        if len(keys) != len(new):
            raise ValueError('Не для всех новых окон найдено сравнение')
        if any(old[k]['label'] != new[k]['label'] for k in keys):
            raise ValueError('Разные метки у сравниваемых окон')
        paired[subject] = {'matched_frames': len(keys),
            'old_raw_accuracy': sum(old[k]['prediction']==old[k]['label'] for k in keys)/len(keys),
            'new_raw_accuracy': sum(new[k]['prediction']==new[k]['label'] for k in keys)/len(keys)}
    (ROOT/'results/study/paired_comparison.json').write_text(json.dumps(paired, indent=2)+'\n')
    print(json.dumps(paired, indent=2))


if __name__ == '__main__':
    main()
