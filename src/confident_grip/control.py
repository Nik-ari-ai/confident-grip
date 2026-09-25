"""Проверка решения классификатора перед передачей команды кисти."""

from dataclasses import dataclass
import math


@dataclass
class Decision:
    command: str | None
    inhibit: bool
    reason: str
    score: float


class GripControl:
    def __init__(self, labels, rest="rest", threshold=0.8, margin=0.2,
                 confirmations=2, max_gap_ms=30):
        if len(labels) < 2 or len(set(labels)) != len(labels) or rest not in labels:
            raise ValueError("Нужны разные классы, включая покой")
        if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("Пороги должны быть от 0 до 1")
        if not isinstance(confirmations, int) or confirmations < 1:
            raise ValueError("Число подтверждений должно быть целым и положительным")
        if not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
            raise ValueError("Интервал между решениями должен быть положительным")
        self.labels = tuple(labels)
        self.threshold = threshold
        self.margin = margin
        self.confirmations = confirmations
        self.max_gap_ms = max_gap_ms
        self.selected = rest
        self.pending = None
        self.count = 0
        self.last_time = None
        self.blocked = True

    def _block(self, reason, score=0.0):
        self.pending = None
        self.count = 0
        self.blocked = True
        return Decision(None, True, reason, score)

    def update(self, scores, time_ms, signal_ok=True):
        """scores идут в порядке labels; signal_ok приходит от проверки ЭМГ.

        inhibit означает запрет выполнения движения. Его должен обрабатывать
        отдельный контроллер привода. Сохранённый selected — только состояние,
        его нельзя повторно отправлять на мотор при inhibit=True.
        """
        if not math.isfinite(time_ms):
            return self._block("неверное время")
        if self.last_time is not None and time_ms <= self.last_time:
            return self._block("время не увеличивается")
        gap = self.last_time is not None and time_ms - self.last_time > self.max_gap_ms
        self.last_time = time_ms
        if gap:
            self._block("пропуск данных")
        if not signal_ok:
            return self._block("плохой сигнал")
        if len(scores) != len(self.labels):
            return self._block("неверное число классов")
        if any(not math.isfinite(x) or x < 0 or x > 1 for x in scores):
            return self._block("неверные оценки модели")
        if not math.isclose(sum(scores), 1.0, abs_tol=1e-5):
            return self._block("оценки не нормированы")

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best, second = order[:2]
        score = float(scores[best])
        label = self.labels[best]
        if score < self.threshold:
            return self._block("низкая уверенность", score)
        if score - scores[second] < self.margin:
            return self._block("два похожих решения", score)

        # После отказа повторно подтверждаем даже прежний хват.
        if label == self.selected and not self.blocked:
            return Decision(label, False, "принято", score)
        if label == self.pending:
            self.count += 1
        else:
            self.pending = label
            self.count = 1
        if self.count < self.confirmations:
            return Decision(None, True, "ожидание подтверждения", score)

        self.selected = label
        self.pending = None
        self.count = 0
        self.blocked = False
        return Decision(label, False, "принято", score)
