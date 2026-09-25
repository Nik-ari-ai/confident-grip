"""Небольшой искусственный пример и чтение упражнения C из NinaPro DB2."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt


LABELS = ("rest", "cylindrical", "pinch", "lateral", "hook", "spherical")
RUSSIAN = ("Покой", "Цилиндрический", "Точный", "Боковой", "Крюк", "Сферический")
# Exercise C: 2, 15, 17, 3, 10. В DB2 перед C идут 17 движений B.
DB2_CLASSES = {19: 1, 32: 2, 34: 3, 20: 4, 27: 5}


@dataclass
class Trial:
    signal: np.ndarray
    labels: np.ndarray
    repetition: int
    name: str
    onset: int
    stop: int
    fs: int = 2000


def synthetic(seed=42):
    """Шум с заданными амплитудами каналов. Не модель физиологии пациента."""
    rng = np.random.default_rng(seed)
    activation = np.array([
        [.08] * 8,
        [.80, .72, .45, .40, .42, .48, .70, .75],
        [.42, .38, .78, .72, .40, .35, .40, .45],
        [.50, .68, .62, .45, .48, .40, .45, .52],
        [.68, .42, .42, .45, .72, .68, .50, .45],
        [.60, .58, .55, .55, .55, .58, .60, .60],
    ])
    fs = 2000
    sos = butter(4, [20, 250], fs=fs, btype="bandpass", output="sos")
    trials = []
    for repetition in range(1, 7):
        gain = rng.uniform(.65, 1.35, 8)
        for label in range(1, 6):
            y = np.zeros(6000, dtype=int)
            y[1000:5000] = label
            x = sosfilt(sos, rng.normal(size=(len(y), 8)), axis=0)
            x *= 40 + 260 * activation[y] * gain
            t = np.arange(len(y)) / fs
            x += (30*np.sin(2*np.pi*50*t) + 20*np.sin(2*np.pi*.4*t))[:, None]
            x += rng.normal(0, 14, x.shape)
            trials.append(Trial(x, y, repetition, f"synthetic_r{repetition}_g{label}", 1000, 5000))
    return trials


def load_db2(path):
    mat = loadmat(path, variable_names=["emg", "restimulus", "rerepetition", "exercise"])
    x = np.asarray(mat["emg"], dtype=float)
    y = np.asarray(mat["restimulus"]).ravel().astype(int)
    rep = np.asarray(mat["rerepetition"]).ravel().astype(int)
    if x.ndim != 2 or x.shape[1] != 12 or len(x) != len(y) or len(y) != len(rep):
        raise ValueError("Ожидается DB2: emg N×12 и две метки длины N")
    if not np.isfinite(x).all():
        raise ValueError("В исходном файле есть пропущенные или неверные отсчёты")
    exercise = int(np.asarray(mat.get("exercise", [[2]])).ravel()[0])
    if exercise != 2:
        raise ValueError("Нужно упражнение E2 (группа C: хваты), не E1")
    # Встречаются экспортированные файлы с локальными номерами 1..23.
    if y.max() <= 23:
        y = np.where(y > 0, y + 17, 0)
    if not set(DB2_CLASSES).issubset(set(y)):
        raise ValueError("В файле не найдены все выбранные хваты")
    x = x[:, :8]
    changes = np.r_[0, np.flatnonzero(np.diff(y)) + 1, len(y)]
    trials = []
    for left, right in zip(changes[:-1], changes[1:]):
        original = int(y[left])
        if original not in DB2_CLASSES:
            continue
        # Берём только соседний покой, не склеиваем удалённые движения.
        start, end = left, right
        while start > max(0, left - 1000) and y[start - 1] == 0:
            start -= 1
        while end < min(len(y), right + 1000) and y[end] == 0:
            end += 1
        active_reps = np.unique(rep[left:right])
        if len(active_reps) != 1 or active_reps[0] not in range(1, 7):
            raise ValueError("Неоднозначная разметка номера повторения")
        repetition = int(active_reps[0])
        mapped = np.zeros(end - start, dtype=int)
        mapped[left-start:right-start] = DB2_CLASSES[original]
        trials.append(Trial(x[start:end].copy(), mapped, repetition,
                            f"{Path(path).stem}_r{repetition}_m{original}", left-start, right-start))
    # Короткий промежуток покоя не должен попасть в два разных разбиения.
    # В стандартном DB2 он существенно длиннее двух выбранных полусекунд.
    selected = [(int(a), int(b)) for a, b in zip(changes[:-1], changes[1:]) if int(y[a]) in DB2_CLASSES]
    for (_, previous_end), (next_start, _) in zip(selected, selected[1:]):
        if next_start - previous_end < 2000 and np.all(y[previous_end:next_start] == 0):
            raise ValueError("Соседний покой слишком короткий: нужна неперекрывающаяся разметка")
    if len(trials) != 30:
        raise ValueError(f"Ожидалось 5 хватов × 6 повторений, получено {len(trials)}")
    return trials
