from __future__ import annotations

import unittest

import numpy as np

from algorithm_evaluation.v5_empirical_ktn152_run import (
    _integerize_state_labels,
    aggregate_chromosome_distances,
    chromosome_slices,
)


class EmpiricalKtn152DistanceTest(unittest.TestCase):
    def test_chromosome_slices_preserve_blocks(self) -> None:
        rows = [
            {"bin_id": "b1", "chromosome": "chr1"},
            {"bin_id": "b2", "chromosome": "chr1"},
            {"bin_id": "b3", "chromosome": "chr2"},
        ]
        self.assertEqual(
            chromosome_slices(rows),
            (("chr1", 0, 2), ("chr2", 2, 3)),
        )

    def test_chromosome_reentry_is_rejected(self) -> None:
        rows = [
            {"bin_id": "b1", "chromosome": "chr1"},
            {"bin_id": "b2", "chromosome": "chr2"},
            {"bin_id": "b3", "chromosome": "chr1"},
        ]
        with self.assertRaisesRegex(ValueError, "re-enters"):
            chromosome_slices(rows)

    def test_directed_costs_are_summed_before_one_minimum(self) -> None:
        labels = ["a", "b"]
        results = [
            {
                "ids": ["a", "b"],
                "directed_matrix": [[0, 10], [1, 0]],
            },
            {
                # Exercise explicit row realignment as well as aggregation.
                "ids": ["b", "a"],
                "directed_matrix": [[0, 10], [1, 0]],
            },
        ]
        directed, minimum = aggregate_chromosome_distances(results, labels)
        np.testing.assert_array_equal(directed, [[0, 11], [11, 0]])
        np.testing.assert_array_equal(minimum, [[0, 11], [11, 0]])

        # Taking a separate minimum on each chromosome would incorrectly give 2.
        self.assertNotEqual(float(minimum[0, 1]), 2.0)

    def test_external_state_labels_are_integerized_without_losing_recurrence(self) -> None:
        payload = {
            "schema_version": "ctbf-v5-reconstruction-input-v1",
            "levels": [
                {
                    "biopsy_level": 0,
                    "generation": 0,
                    "states": [
                        {"state_label": "sha_b", "cnp": [2, 3]},
                        {"state_label": "sha_a", "cnp": [2, 2]},
                    ],
                },
                {
                    "biopsy_level": 1,
                    "generation": 2,
                    "states": [{"state_label": "sha_a", "cnp": [2, 2]}],
                },
            ],
        }
        internal, external_by_internal, mapping = _integerize_state_labels(payload)
        self.assertEqual(external_by_internal, {0: "sha_a", 1: "sha_b"})
        self.assertEqual(
            [state["state_label"] for state in internal["levels"][0]["states"]],
            [1, 0],
        )
        self.assertEqual(internal["levels"][1]["states"][0]["state_label"], 0)
        self.assertEqual(len(mapping["entries"]), 2)
        self.assertEqual(payload["levels"][0]["states"][0]["state_label"], "sha_b")


if __name__ == "__main__":
    unittest.main()
