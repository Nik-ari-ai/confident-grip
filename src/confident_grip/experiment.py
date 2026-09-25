"""Обучение двух моделей и последовательная проверка команд на отложенных повторах."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .control import GripControl
from .data import LABELS, RUSSIAN, DB2_CLASSES, synthetic, load_db2
from .processing import SignalFilter, features, prepare, fit_quality, signal_quality


SPLIT = {"train": [1, 3, 4], "validation": [2], "test": [5, 6]}


def train_models(prepared):
    x = np.concatenate([p["features"][p["train_ok"]] for p in prepared])
    y = np.concatenate([p["labels"][p["train_ok"]] for p in prepared])
    groups = np.concatenate([np.full(p["train_ok"].sum(), p["trial"].repetition) for p in prepared])
    folds = list(GroupKFold(3).split(x, y, groups))
    models = {}
    for name, classifier in [
        ("LDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("SVM", SVC(C=10, gamma="scale")),
    ]:
        base = make_pipeline(StandardScaler(), classifier)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=folds)
        model.fit(x, y)
        models[name] = model
    return models


def predict(model, prepared, quality_limit):
    result = []
    for p in prepared:
        result.append({**p, "scores": model.predict_proba(p["features"]),
                       "quality": signal_quality(p, quality_limit)})
    return result


def evaluate(predictions, threshold=0., confirmations=1, margin=0.):
    truth, raw, commands, rows = [], [], [], []
    wrong_switches = 0
    duration = 0.
    response = []
    missed = 0
    stable_response = []
    stable_missed = 0
    for p in predictions:
        control = GripControl(range(6), rest=0, threshold=threshold, margin=margin,
                              confirmations=confirmations, max_gap_ms=30)
        previous = 0
        accepted = []
        for i, (time_ms, scores, ok) in enumerate(zip(p["time_ms"], p["scores"], p["quality"])):
            decision = control.update(scores, time_ms, bool(ok))
            command = -1 if decision.command is None else int(decision.command)
            label = int(p["labels"][i])
            raw_label = int(np.argmax(scores))
            if command >= 0 and command != previous:
                wrong_switches += command != label
                previous = command
            accepted.append(command)
            rows.append([p["trial"].name, float(time_ms), label, raw_label, command,
                         bool(decision.inhibit), decision.reason])
        trial = p["trial"]
        onset_ms, stop_ms = trial.onset * 1000 / trial.fs, trial.stop * 1000 / trial.fs
        target = int(trial.labels[trial.onset])
        matches = [t for t, command in zip(p["time_ms"], accepted)
                   if onset_ms <= t < stop_ms and command == target]
        if matches:
            response.append(float(matches[0] - onset_ms))
        else:
            missed += 1
        # Решение считается устойчивым после 100 мс подряд правильных команд.
        # Время ставим в конце подтверждённого участка, без заглядывания вперёд.
        run_start = None
        stable = None
        for t, command in zip(p["time_ms"], accepted):
            if onset_ms <= t < stop_ms and command == target:
                if run_start is None:
                    run_start = t
                if t - run_start >= 100:
                    stable = float(t - onset_ms)
                    break
            else:
                run_start = None
        if stable is None:
            stable_missed += 1
        else:
            stable_response.append(stable)
        duration += (p["time_ms"][-1] - p["time_ms"][0] + 10) / 1000
        truth.extend(p["labels"].tolist())
        raw.extend(np.argmax(p["scores"], axis=1).tolist())
        commands.extend(accepted)
    y, raw, commands = np.asarray(truth), np.asarray(raw), np.asarray(commands)
    issued = commands >= 0
    wrong = issued & (commands != y)
    result = {
        "frames": len(y), "raw_accuracy": float(accuracy_score(y, raw)),
        "raw_macro_f1": float(f1_score(y, raw, labels=range(6), average="macro", zero_division=0)),
        "accepted_fraction": float(issued.mean()),
        "wrong_fraction_all": float(wrong.mean()),
        "wrong_fraction_accepted": float(wrong.sum() / issued.sum()) if issued.any() else None,
        "wrong_switches": int(wrong_switches),
        "wrong_switches_per_minute": float(wrong_switches / duration * 60),
        "response_median_ms": float(np.median(response)) if response else None,
        "response_p95_ms": float(np.quantile(response, .95)) if response else None,
        "stable_response_median_ms": float(np.median(stable_response)) if stable_response else None,
        "stable_response_p95_ms": float(np.quantile(stable_response, .95)) if stable_response else None,
        "stable_missed_movements": stable_missed,
        "missed_movements": missed, "movements": len(predictions),
        "response_within_100ms_fraction": float(sum(t <= 100 for t in response) / len(predictions)),
    }
    return result, rows


def choose_settings(validation):
    candidates = []
    for threshold in (0., .5, .7, .85, .95):
        for confirmations in (1, 2, 3):
            result, _ = evaluate(validation, threshold, confirmations)
            # Заранее заданная функция выбора: ошибки, отказы и пропущенные движения.
            loss = (result["wrong_fraction_all"] + .2 * (1 - result["accepted_fraction"])
                    + .1 * result["missed_movements"] / result["movements"])
            candidates.append({"threshold": threshold, "confirmations": confirmations,
                               "loss": loss, "metrics": result})
    best = min(candidates, key=lambda c: (c["loss"], c["confirmations"], c["threshold"]))
    return {k: best[k] for k in ("threshold", "confirmations")}, candidates


def benchmark(model, trial, settings, quality_limit, window_ms=50):
    filt = SignalFilter(trial.fs, trial.signal.shape[1])
    width, step = round(trial.fs * window_ms / 1000), round(trial.fs * .01)
    ring = np.zeros((width, trial.signal.shape[1]))
    raw_ring = np.zeros_like(ring)
    control = GripControl(range(6), rest=0, **({"margin": 0} | settings))
    times = []
    for index, start in enumerate(range(0, min(len(trial.signal)-step, 6000), step)):
        t0 = perf_counter()
        clean = filt.feed(trial.signal[start:start+step])
        t1 = perf_counter()
        ring = np.vstack([ring[step:], clean])
        raw_ring = np.vstack([raw_ring[step:], trial.signal[start:start+step]])
        vector = features(ring)
        quality = signal_quality({"raw_sd": raw_ring.std(axis=0)[None, :]}, quality_limit)[0]
        t2 = perf_counter()
        scores = model.predict_proba(vector[None, :])[0]
        t3 = perf_counter()
        control.update(scores, index*10, bool(quality))
        t4 = perf_counter()
        if index >= 20:
            times.append(np.diff([t0, t1, t2, t3, t4]) * 1000)
    times = np.array(times)
    result = {key: float(value) for key, value in zip(
        ["filter_p95_ms", "features_p95_ms", "model_p95_ms", "decision_p95_ms"],
        np.quantile(times, .95, axis=0))}
    result["compute_total_p95_ms"] = float(np.quantile(times.sum(axis=1), .95))
    result["note"] = "Время вычислений на компьютере; без сбора окна, связи и привода"
    return result


def plot_example(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(row[0] for row in rows))
    # Первый тестовый повтор, выбор не зависит от получившегося качества.
    shown = [row for row in rows if row[0] == names[0]]
    t = np.array([row[1] for row in shown]) / 1000
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.step(t, [row[2] for row in shown], where="post", label="Метка записи", lw=2, color="#246345")
    ax.step(t, [row[3] for row in shown], where="post", label="Ответ модели", alpha=.6, color="#a4662b")
    ax.scatter(t, [row[4] for row in shown], label="После проверки", s=7, color="#254c7a")
    ax.set_yticks(range(-1, 6), ["Отказ", *RUSSIAN])
    ax.set_xlabel("Время от начала фрагмента, с")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="S1_E2_A1.mat из NinaPro DB2")
    parser.add_argument("--output", type=Path, default=Path("results/synthetic"))
    args = parser.parse_args()
    trials = load_db2(args.data) if args.data else synthetic()
    prepared = [prepare(trial) for trial in trials]
    parts = {name: [p for p in prepared if p["trial"].repetition in reps] for name, reps in SPLIT.items()}
    models = train_models(parts["train"])
    quality_limit = fit_quality(parts["train"])
    report = {
        "source": "NinaPro DB2, один участник" if args.data else "Искусственный сигнал, seed=42",
        "file": args.data.name if args.data else None,
        "sha256": hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data else None,
        "split": SPLIT, "window_ms": 50, "step_ms": 10,
        "class_mapping": DB2_CLASSES if args.data else None,
        "labels": LABELS, "channels_zero_based": list(range(8)),
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "models": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        validation = predict(model, parts["validation"], quality_limit)
        settings, candidates = choose_settings(validation)
        testing = predict(model, parts["test"], quality_limit)
        variants = {"baseline": {"threshold": 0., "confirmations": 1},
                    "threshold_only": {"threshold": settings["threshold"], "confirmations": 1},
                    "confirmation_only": {"threshold": 0., "confirmations": settings["confirmations"]},
                    "combined": settings}
        item = {"settings": settings, "validation_candidates": candidates, "test": {}}
        for variant, params in variants.items():
            metrics, rows = evaluate(testing, **params)
            item["test"][variant] = metrics
            with (args.output/f"{name.lower()}_{variant}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["trial", "time_ms", "label", "prediction", "command", "inhibit", "reason"])
                writer.writerows(rows)
            if variant == "combined":
                plot_example(rows, args.output/f"{name.lower()}_trace.png")
        y = np.concatenate([p["labels"] for p in testing])
        raw = np.concatenate([p["scores"].argmax(axis=1) for p in testing])
        item["confusion_matrix"] = confusion_matrix(y, raw, labels=range(6)).tolist()
        item["timing"] = benchmark(model, parts["test"][0]["trial"], settings, quality_limit)
        report["models"][name] = item
        print(name, settings, json.dumps(item["test"], ensure_ascii=False), flush=True)
    (args.output/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
