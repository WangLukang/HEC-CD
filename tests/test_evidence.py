import unittest

import numpy as np
import torch

try:
    from hec_cd import evidence
except ModuleNotFoundError as error:
    if error.name not in {"hec_cd", "hec_cd.evidence"}:
        raise
    evidence = None


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evidence, "The public evidence API has not been implemented")

    def test_semantic_candidates_include_030_and_require_20_pixels(self):
        first = np.zeros((10, 12), np.float32)
        first[1:5, 1:6] = 0.30
        first[6:10, 7:12] = 0.299
        second = np.zeros_like(first)
        second[1:5, 7:12] = 0.8
        second[1, 7] = 0
        empty = np.zeros((0, *first.shape), bool)
        masks = evidence.proposals(empty, empty, first, second)
        self.assertEqual(len(masks), 1)
        self.assertEqual(int(masks[0].sum()), 20)
        self.assertEqual(masks[0].dtype, np.bool_)

    def test_iou_deduplication_is_inclusive_at_085(self):
        masks = np.zeros((3, 5, 8), bool)
        masks[0] = True
        masks[1].flat[:34] = True
        masks[2].flat[:33] = True
        zero = np.zeros((5, 8), np.float32)
        actual = evidence.proposals(masks, masks[:0], zero, zero)
        self.assertEqual([int(mask.sum()) for mask in actual], [40, 33])

    def test_proposals_keep_largest_first_and_cap_at_96(self):
        masks = np.zeros((97, 4, 5 * 97), bool)
        for index in range(97):
            masks[index, :, 5 * index:5 * (index + 1)] = True
        zero = np.zeros(masks.shape[1:], np.float32)
        actual = evidence.proposals(masks, masks[:0], zero, zero)
        self.assertEqual(len(actual), 96)
        np.testing.assert_array_equal(actual[0], masks[0])
        self.assertFalse(any(np.array_equal(mask, masks[96]) for mask in actual))

    def test_mask_pooling_precedes_cosine_distance(self):
        mask = np.ones((4, 8), bool)
        first = torch.tensor([[[1., 0.]], [[0., 1.]]])
        second = torch.tensor([[[0., 1.]], [[1., 0.]]])
        actual = evidence.mask_distances([mask], [first], [second])
        np.testing.assert_allclose(actual, [0], atol=1e-6)
        self.assertEqual(actual.dtype, np.float32)

    def test_cosine_distance_is_clipped_and_empty_masks_are_supported(self):
        mask = np.ones((5, 5), bool)
        first = torch.tensor([[[1.]], [[0.]]])
        np.testing.assert_array_equal(evidence.mask_distances([mask], [first], [-first]), [1])
        actual = evidence.mask_distances([], [first], [first])
        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, np.float32)

    def test_object_evidence_uses_normalized_distance_sqrt_and_max_projection(self):
        masks = np.zeros((2, 5, 8), bool)
        masks[0, :, :5] = True
        masks[1, :, 3:] = True
        first_probability = np.full((5, 8), 0.25, np.float32)
        second_probability = np.zeros((5, 8), np.float32)
        first_features = [torch.tensor([[[1.]], [[0.]]])]
        second_features = [torch.tensor([[[0.9]], [[np.sqrt(0.19)]]], dtype=torch.float32)]
        actual = evidence.object_evidence(masks, masks[:0], first_probability,
                                          second_probability, first_features, second_features)
        np.testing.assert_allclose(actual, np.sqrt(0.125), atol=2e-7)
        self.assertEqual(actual.dtype, np.float32)

    def test_object_semantic_delta_is_difference_of_mask_means(self):
        masks = np.ones((1, 4, 5), bool)
        first_probability = np.zeros((4, 5), np.float32)
        first_probability[:, :2] = 0.25
        second_probability = np.zeros((4, 5), np.float32)
        second_probability[:, 3:] = 0.25
        first = [torch.tensor([[[1.]], [[0.]]])]
        second = [torch.tensor([[[0.]], [[1.]]])]
        actual = evidence.object_evidence(masks, masks[:0], first_probability,
                                          second_probability, first, second)
        np.testing.assert_array_equal(actual, np.zeros((4, 5)))

    def test_normalized_object_distance_saturates_at_one(self):
        masks = np.ones((1, 4, 5), bool)
        first_probability = np.full((4, 5), 0.25, np.float32)
        second_probability = np.zeros((4, 5), np.float32)
        first = [torch.tensor([[[1.]], [[0.]]])]
        second = [torch.tensor([[[0.]], [[1.]]])]
        actual = evidence.object_evidence(masks, masks[:0], first_probability,
                                          second_probability, first, second)
        np.testing.assert_array_equal(actual, np.full((4, 5), 0.5))

    def test_no_proposals_produces_zero_evidence(self):
        empty = np.zeros((0, 4, 5), bool)
        zero = np.zeros((4, 5), np.float32)
        features = [torch.zeros((2, 1, 1))]
        np.testing.assert_array_equal(evidence.object_evidence(empty, empty, zero, zero,
                                                              features, features), zero)


if __name__ == "__main__":
    unittest.main()
