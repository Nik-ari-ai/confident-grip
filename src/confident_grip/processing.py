"""Причинная фильтрация и признаки из исходного прототипа, с исправлениями."""

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt


class SignalFilter:
    def __init__(self, fs, channels):
        band = butter(4, [20, 450], btype="bandpass", fs=fs, output="sos")
        b, a = iirnotch(50, Q=30, fs=fs)
        self.sos = np.vstack([band, tf2sos(b, a)])
        self.state = np.zeros((len(self.sos), 2, channels))

    def feed(self, samples):
        clean, self.state = sosfilt(self.sos, samples, axis=0, zi=self.state)
        return clean


def features(window):
    rms = np.sqrt(np.mean(window ** 2, axis=0))
    mav = np.mean(np.abs(window), axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    threshold = .01 * rms + np.finfo(float).eps
    zc = np.sum((window[:-1] * window[1:] < 0) &
                (np.abs(window[1:] - window[:-1]) >= threshold), axis=0)
    left = window[1:-1] - window[:-2]
    right = window[1:-1] - window[2:]
    # Оба порога имеют единицы амплитуды, а не амплитуды в квадрате.
    ssc = np.sum((left * right > 0) &
                 (np.maximum(np.abs(left), np.abs(right)) >= threshold), axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])


def prepare(trial, window_ms=50, step_ms=10):
    width = round(trial.fs * window_ms / 1000)
    step = round(trial.fs * step_ms / 1000)
    if width < 3 or step < 1:
        raise ValueError("Слишком короткое окно или шаг")
    clean = SignalFilter(trial.fs, trial.signal.shape[1]).feed(trial.signal)
    ends = np.arange(width, len(clean) + 1, step)
    feat = np.array([features(clean[end-width:end]) for end in ends])
    raw_sd = np.array([trial.signal[end-width:end].std(axis=0) for end in ends])
    # На тесте метка берётся в конце окна: переходы не удаляются.
    labels = trial.labels[ends - 1]
    pure = np.array([np.all(trial.labels[end-width:end] == labels[i]) for i, end in enumerate(ends)])
    # Первые 200 мс исключаем только из обучения, чтобы не учить пусковой переход фильтра.
    train_ok = pure & (ends >= round(.2 * trial.fs))
    return {"trial": trial, "features": feat, "labels": labels, "raw_sd": raw_sd,
            "train_ok": train_ok, "time_ms": ends * 1000 / trial.fs}


def fit_quality(prepared):
    values = np.concatenate([p["raw_sd"][p["train_ok"]] for p in prepared])
    return np.quantile(values, .999, axis=0) * 5 + np.finfo(float).eps


def signal_quality(prepared, upper):
    sd = prepared["raw_sd"]
    return np.all(np.isfinite(sd) & (sd > 1e-12) & (sd <= upper), axis=1)
