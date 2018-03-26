from unittest import TestCase
from ..entity_overlapping_score import entity_overlapping_score


class OverlappingScoreTestCase(TestCase):

    def test_evaluate(self):
        test_cases = [
            (
                {
                    "entity_prediction": [
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "DONT_CARE", "value": "2"},
                        {"name": "DONT_CARE", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                        {"name": "DONT_CARE", "value": "1"},
                        {"name": "DONT_CARE", "value": "2"},
                        {"name": "DONT_CARE", "value": "3"},
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
                        {"name": "1", "value": "1"},
                        {"name": "2", "value": "2"},
                        {"name": "3", "value": "3"},
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
                result = entity_overlapping_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
