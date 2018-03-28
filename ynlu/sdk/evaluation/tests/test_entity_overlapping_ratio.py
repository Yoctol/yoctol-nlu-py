from unittest import TestCase
from ..entity_overlapping_score import (
    single__entity_overlapping_score,
    entity_overlapping_score,
)


class OverlappingScoreTestCase(TestCase):

    def test_single__entity_overlapping_score_different_length(self):
        with self.assertRaises(ValueError):
            single__entity_overlapping_score(
                utterance="12",
                entity_prediction=[
                    {"value": "1", "entity": "a"},
                    {"value": "2", "entity": "b"},
                ],
                y_true=["a", "b", "c"],
            )

    def test_single__entity_overlapping_score(self):
        test_cases = [
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["4", "5", "6"],
                    "wrong_penalty_rate": 2.0,
                },
                -1.0,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["4", "DONT_CARE", "6"],
                    "wrong_penalty_rate": 2.0,
                },
                -0.666666666667,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["4", "2", "6"],
                    "wrong_penalty_rate": 2.0,
                },
                -0.33333333333333,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["DONT_CARE", "DONT_CARE", "DONT_CARE"],
                    "wrong_penalty_rate": 2.0,
                },
                0.0,
            ),
            (
                {

                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "DONT_CARE", "value": "2"},
                        {"entity": "DONT_CARE", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["DONT_CARE", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                0.0,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["DONT_CARE", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                0.6666666666666667,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["5", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                0.3333333333333333,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "DONT_CARE", "value": "1"},
                        {"entity": "DONT_CARE", "value": "2"},
                        {"entity": "DONT_CARE", "value": "3"},
                    ],
                    "utterance": "123",

                    "y_true": ["DONT_CARE", "DONT_CARE", "DONT_CARE"],
                    "wrong_penalty_rate": 2.0,
                },
                1.0,
            ),
            (
                {
                    "entity_prediction": [
                        {"entity": "1", "value": "1"},
                        {"entity": "2", "value": "2"},
                        {"entity": "3", "value": "3"},
                    ],
                    "utterance": "123",
                    "y_true": ["1", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                1.0,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = single__entity_overlapping_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_entity_overlapping_score_different_amount(self):
        with self.assertRaises(ValueError):
            entity_overlapping_score(
                utterances=["123", "345"],
                entity_predictions=[[{"a": 1}], [{"b": 2}]],
                y_trues=[["a"], ["b"], ["c"]],
            )

    def test_entity_overlapping_score(self):
        result = entity_overlapping_score(
            utterances=["123", "123"],
            entity_predictions=[
                [
                    {"entity": "1", "value": "1"},
                    {"entity": "2", "value": "2"},
                    {"entity": "3", "value": "3"},
                ],
                [
                    {"entity": "DONT_CARE", "value": "1"},
                    {"entity": "DONT_CARE", "value": "2"},
                    {"entity": "DONT_CARE", "value": "3"},
                ],
            ],
            y_trues=[
                ["5", "2", "3"],
                ["DONT_CARE", "DONT_CARE", "DONT_CARE"],
            ],
        )
        self.assertAlmostEqual(
            (0.33333333333 + 1.0) / 2,
            result,
        )
