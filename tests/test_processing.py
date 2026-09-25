import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat

from confident_grip.data import DB2_CLASSES, load_db2, synthetic
from confident_grip.processing import SignalFilter, prepare, signal_quality
from confident_grip.experiment import SPLIT


class ProcessingTests(unittest.TestCase):
    def test_stream_filter_matches_whole_record(self):
        x = np.random.default_rng(1).normal(size=(2000, 8))
        expected = SignalFilter(2000, 8).feed(x)
        stream = SignalFilter(2000, 8)
        actual = np.vstack([stream.feed(chunk) for chunk in np.array_split(x, 100)])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_filter_cannot_see_future_samples(self):
        x = np.random.default_rng(2).normal(size=(2000, 8))
        changed = x.copy()
        changed[1000:] *= 100
        np.testing.assert_array_equal(SignalFilter(2000, 8).feed(x)[:1000],
                                      SignalFilter(2000, 8).feed(changed)[:1000])

    def test_quality_rejects_dead_channel(self):
        prepared = {"raw_sd": np.array([[1., 1.], [0., 1.]])}
        np.testing.assert_array_equal(signal_quality(prepared, np.array([5., 5.])), [True, False])

    def test_split_has_no_common_repetitions(self):
        self.assertEqual(set(SPLIT["train"]) & set(SPLIT["validation"]), set())
        self.assertEqual(set(SPLIT["train"]) & set(SPLIT["test"]), set())
        self.assertEqual(set(SPLIT["test"]) & set(SPLIT["validation"]), set())

    def test_windows_keep_current_label_at_transition(self):
        trial = synthetic()[0]
        prepared = prepare(trial)
        i = np.flatnonzero(prepared["time_ms"] > trial.onset / 2)[0]
        self.assertEqual(prepared["labels"][i], 1)
        self.assertFalse(prepared["train_ok"][i])

    def test_db2_loader_selects_grasps_without_joining_them(self):
        labels, repetitions = [], []
        for repetition in range(1, 7):
            for original in DB2_CLASSES:
                labels.extend([0]*2000 + [original]*500 + [0]*2000)
                repetitions.extend([0]*2000 + [repetition]*500 + [0]*2000)
        y = np.array(labels)
        x = np.random.default_rng(3).normal(size=(len(y), 12))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"S1_E2_A1.mat"
            savemat(path, {"emg": x, "restimulus": y, "rerepetition": repetitions, "exercise": 2})
            trials = load_db2(path)
        self.assertEqual(len(trials), 30)
        self.assertEqual(trials[0].signal.shape, (2500, 8))
        self.assertEqual(trials[0].onset, 1000)
        self.assertEqual(set(np.unique(t.labels[t.onset:t.stop]).item() for t in trials), set(range(1, 6)))


if __name__ == "__main__":
    unittest.main()
