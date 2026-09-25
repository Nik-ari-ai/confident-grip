"""Встраивает сохранённые команды в автономную HTML-демонстрацию."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    data = {}
    for subject in ('s2', 's3'):
        metrics = json.loads((ROOT/f'results/study/{subject}/metrics.json').read_text())
        for name in ('LDA', 'SVM'):
            trials = {}
            with (ROOT/f'results/study/{subject}/{name.lower()}_combined.csv').open() as stream:
                for row in csv.DictReader(stream):
                    trials.setdefault(row['trial'], []).append([
                        float(row['time_ms']), int(row['label']), int(row['prediction']),
                        int(row['command']), row['reason']])
            data[f'{subject.upper()} / {name}'] = {
                'trials': trials, 'metrics': metrics['models'][name]['test']['combined']}
    template = (ROOT/'scripts/demo_template.html').read_text()
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    (ROOT/'docs/demo.html').write_text(template.replace('/* DATA */ {}', encoded))
    print(ROOT/'docs/demo.html')

if __name__ == '__main__':
    main()
