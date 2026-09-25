import unittest
import numpy as np
from confident_grip.data import Trial
from confident_grip.experiment import evaluate
from confident_grip.study import log_features, select_gate


def prediction(labels, outputs, onset=50):
    times = np.arange(10, len(labels)*10+1, 10, dtype=float)
    scores = np.full((len(labels), 6), .01)
    for i, value in enumerate(outputs):
        scores[i, value] = .95
    trial = Trial(np.ones((len(labels)*10+10, 8)),
                  np.r_[np.zeros(onset, dtype=int), np.ones(len(labels)*10+10-onset, dtype=int)],
                  5, 'test', onset, len(labels)*10+1, fs=1000)
    return dict(trial=trial, time_ms=times, labels=np.asarray(labels), scores=scores,
                quality=np.ones(len(labels), dtype=bool))


class StudyTests(unittest.TestCase):
    def test_log_features_finite_for_silent_channel(self):
        result = log_features(np.zeros((2, 40)))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, 24:], 0.)

    def test_stable_response_does_not_credit_one_lucky_frame(self):
        output = [0]*4 + [1] + [2]*9 + [1]*16
        p = prediction([0]*4 + [1]*26, output)
        metrics, _ = evaluate([p])
        self.assertEqual(metrics['response_median_ms'], 0)
        self.assertEqual(metrics['stable_response_median_ms'], 200)

    def test_missed_movement_remains_in_response_denominator(self):
        good = prediction([0]*4 + [1]*26, [0]*4 + [1]*26)
        missed = prediction([0]*4 + [1]*26, [0]*30)
        metrics, _ = evaluate([good, missed])
        self.assertEqual(metrics['missed_movements'], 1)
        self.assertEqual(metrics['response_within_100ms_fraction'], .5)

    def test_margin_changes_evaluation(self):
        p = prediction([1]*30, [1]*30)
        p['scores'][:] = [.01, .49, .47, .01, .01, .01]
        base, _ = evaluate([p], margin=0)
        gated, _ = evaluate([p], margin=.15)
        self.assertEqual(base['accepted_fraction'], 1)
        self.assertEqual(gated['accepted_fraction'], 0)

    def test_gate_selection_respects_validation_coverage(self):
        p = prediction([0]*4 + [1]*26, [0]*4 + [1]*26)
        params, _, feasible = select_gate([p])
        metrics, _ = evaluate([p], **params)
        self.assertTrue(feasible)
        self.assertGreaterEqual(metrics['accepted_fraction'], .70)

    def test_bad_signal_has_no_feasible_gate(self):
        p = prediction([1]*30, [1]*30)
        p['quality'][:] = False
        _, _, feasible = select_gate([p])
        self.assertFalse(feasible)


if __name__ == '__main__':
    unittest.main()
