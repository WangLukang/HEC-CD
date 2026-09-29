import unittest

import numpy as np

try:
    from hec_cd import core
except ModuleNotFoundError as error:
    if error.name not in {"hec_cd", "hec_cd.core"}:
        raise
    core = None


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, "The public core API has not been implemented")

    def test_quantization_clips_and_rounds_half_to_even(self):
        scores = np.array([-1, 0, 0.5 / 255, 1.5 / 255, 0.5, 1, 2])
        np.testing.assert_array_equal(core.quantize(scores), [0, 0, 0, 2, 128, 255, 255])
        self.assertEqual(core.quantize(scores).dtype, np.uint8)
        np.testing.assert_array_equal(core.quantize(np.array([-1, 84, 300])), [0, 84, 255])

    def test_harmonic_consensus_requires_both_branches(self):
        actual = core.harmonic_consensus([0, 0.25, 0.8], [1, 0.75, 0.4])
        np.testing.assert_allclose(actual, [0, 0.375, 8 / 15], atol=1e-7)
        self.assertEqual(actual.dtype, np.float32)
        with self.assertRaises(ValueError):
            core.harmonic_consensus([0, 1], [1])

    def test_ght_tie_uses_mean_split_then_inclusive_cutoff(self):
        histogram = np.zeros(256, np.int64)
        histogram[0], histogram[10], histogram[20] = 50, 2, 2
        result = core.select_unlabeled_histogram(histogram)
        self.assertEqual(result.selected_u8, 15)
        self.assertEqual(result.ght_split_u8, 14.5)
        self.assertEqual(result.nonzero_count, 4)
        self.assertEqual(result.selected_score_01, 15 / 255)
        self.assertEqual(result.parameters["rho"], 10)
        self.assertAlmostEqual(result.parameters["tau_squared_u8"], 1 / 12)

    def test_ght_empirical_prior_uses_pooled_within_class_variance(self):
        histogram = np.zeros(256, np.int64)
        histogram[[10, 12, 20, 22]] = 1
        result = core.select_unlabeled_histogram(histogram)
        self.assertEqual(result.parameters["otsu_split_u8"], 12)
        self.assertEqual(result.parameters["raw_pooled_variance_u8"], 1)
        self.assertEqual(result.parameters["nu"], 40)
        self.assertAlmostEqual(result.parameters["shrinkage_weight"], 10 / 11)

    def test_empty_and_constant_support_fallbacks(self):
        histogram = np.zeros(256, np.int64)
        histogram[0] = 100
        result = core.select_unlabeled_histogram(histogram)
        self.assertEqual(result.selected_u8, 256)
        self.assertFalse(core.postprocess(np.full((20, 20), 255, np.uint8), 256).any())
        histogram[93] = 2
        result = core.select_unlabeled_histogram(histogram)
        self.assertEqual(result.selected_u8, 93)
        self.assertEqual(result.nonzero_count, 2)
        self.assertIsNone(result.ght_split_u8)

    def test_invalid_histograms_are_rejected(self):
        for histogram in (np.zeros(10), np.full(256, -1), np.full(256, np.nan)):
            with self.subTest(histogram=histogram):
                with self.assertRaises(ValueError):
                    core.select_unlabeled_histogram(histogram)

    def test_cross_opening_uses_inclusive_cutoff(self):
        score = np.zeros((24, 24), np.uint8)
        score[2:16, 2:16] = 128
        expected = score > 0
        for row, col in ((2, 2), (2, 15), (15, 2), (15, 15)):
            expected[row, col] = False
        score[20, 20] = 128  # An isolated pixel does not survive the cross opening.
        actual = core.postprocess(score, 128)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(core.postprocess(score, 129).any())

    def test_postprocess_filters_by_area_after_opening(self):
        score = np.zeros((40, 40), np.uint8)
        score[2:16, 2:16] = 255  # 196 pixels minus four corners = 192.
        score[20:33, 20:35] = 255  # 195 pixels minus four corners = 191.
        prediction = core.postprocess(score, 128)
        self.assertEqual(int(prediction.sum()), 192)
        self.assertTrue(prediction[8, 8])
        self.assertFalse(prediction[25, 25])
        self.assertEqual(prediction.dtype, np.bool_)

    def test_postprocess_does_not_merge_diagonally_touching_regions(self):
        rows, cols = np.indices((40, 40))
        # Each diamond has 181 pixels and survives the cross opening unchanged.
        first = np.abs(rows - 12) + np.abs(cols - 12) <= 9
        second = np.abs(rows - 22) + np.abs(cols - 22) <= 9
        score = ((first | second) * 255).astype(np.uint8)
        self.assertFalse(core.postprocess(score, 128).any())

    def test_metrics_use_global_counts(self):
        actual = core.metrics_from_counts(80, 20, 40, 860)
        for name, expected in {"precision": 0.8, "recall": 2 / 3,
                               "f1": 8 / 11, "iou": 4 / 7, "oa": 0.94}.items():
            self.assertAlmostEqual(actual[name], expected)


if __name__ == "__main__":
    unittest.main()
