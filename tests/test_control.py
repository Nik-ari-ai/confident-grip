import unittest
from confident_grip.control import GripControl


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.control = GripControl(["rest", "pinch", "hook"])

    def test_single_wrong_frame_does_not_change_grip(self):
        self.control.update([0.02, 0.96, 0.02], 0)
        self.assertEqual(self.control.update([0.02, 0.96, 0.02], 10).command, "pinch")
        result = self.control.update([0.02, 0.02, 0.96], 20)
        self.assertTrue(result.inhibit)
        self.assertIsNone(result.command)
        self.assertEqual(self.control.selected, "pinch")

    def test_bad_signal_does_not_reuse_previous_motor_command(self):
        self.control.update([0.02, 0.96, 0.02], 0)
        self.control.update([0.02, 0.96, 0.02], 10)
        result = self.control.update([0.02, 0.96, 0.02], 20, signal_ok=False)
        self.assertIsNone(result.command)
        self.assertTrue(result.inhibit)
        self.assertTrue(self.control.update([0.02, 0.96, 0.02], 30).inhibit)
        self.assertEqual(self.control.update([0.02, 0.96, 0.02], 40).command, "pinch")

    def test_gap_cancels_confirmation(self):
        self.control.update([0.02, 0.96, 0.02], 0)
        self.assertIsNone(self.control.update([0.02, 0.96, 0.02], 100).command)
        self.assertEqual(self.control.update([0.02, 0.96, 0.02], 110).command, "pinch")

    def test_invalid_input_blocks_command(self):
        for scores in ([float("nan"), 0.5, 0.5], [1, 1, 1], [0.5, 0.5]):
            with self.subTest(scores=scores):
                self.assertTrue(GripControl(["rest", "pinch", "hook"]).update(scores, 0).inhibit)

    def test_low_confidence_and_ambiguous_scores_are_rejected(self):
        self.assertTrue(self.control.update([0.3, 0.4, 0.3], 0).inhibit)
        control = GripControl(["rest", "pinch", "hook"], threshold=0.4, margin=0.2)
        self.assertEqual(control.update([0.1, 0.46, 0.44], 0).reason, "два похожих решения")

    def test_rest_requires_confirmation(self):
        self.control.update([0.02, 0.96, 0.02], 0)
        self.control.update([0.02, 0.96, 0.02], 10)
        self.assertIsNone(self.control.update([0.96, 0.02, 0.02], 20).command)
        self.assertEqual(self.control.update([0.96, 0.02, 0.02], 30).command, "rest")

    def test_duplicate_timestamp_is_not_a_confirmation(self):
        self.control.update([0.02, 0.96, 0.02], 0)
        self.assertTrue(self.control.update([0.02, 0.96, 0.02], 0).inhibit)
        self.assertTrue(self.control.update([0.02, 0.96, 0.02], 10).inhibit)

    def test_confident_persistent_error_is_not_detected(self):
        self.control.update([0.02, 0.02, 0.96], 0)
        self.assertEqual(self.control.update([0.02, 0.02, 0.96], 10).command, "hook")


class ConsecutiveConfirmationTests(unittest.TestCase):
    """Смена хвата допускается только после N одинаковых ответов подряд."""

    LABELS = ["rest", "A", "B"]

    @staticmethod
    def onehot(i):
        return [1.0 if j == i else 0.0 for j in range(3)]

    def run_sequence(self, seq, confirmations):
        control = GripControl(self.LABELS, threshold=0, margin=0,
                              confirmations=confirmations)
        return [control.update(self.onehot(i), (t + 1) * 10).command
                for t, i in enumerate(seq)]

    def test_review_sequence_a_a_b_a_b(self):
        # Промежуточный A прерывает подтверждение B.
        commands = self.run_sequence([1, 1, 2, 1, 2], confirmations=2)
        self.assertEqual(commands, [None, "A", None, "A", None])

    def test_exhaustive_short_sequences(self):
        import itertools
        for confirmations in (2, 3):
            for length in range(1, 9):
                for seq in itertools.product([0, 1, 2], repeat=length):
                    commands = self.run_sequence(seq, confirmations)
                    selected = "rest"
                    for t, command in enumerate(commands):
                        if command is not None and command != selected:
                            label = self.LABELS[seq[t]]
                            self.assertEqual(command, label)
                            window = seq[max(0, t - confirmations + 1):t + 1]
                            self.assertEqual(
                                len(window), confirmations, (seq, t))
                            self.assertTrue(
                                all(self.LABELS[i] == label for i in window),
                                (confirmations, seq, t, commands))
                        if command is not None:
                            selected = command


if __name__ == "__main__":
    unittest.main()
