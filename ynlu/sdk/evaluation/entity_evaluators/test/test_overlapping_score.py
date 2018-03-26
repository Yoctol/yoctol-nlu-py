from unittest import TestCase
from ..overlapping_score import OverlappingScore


class OverlappingScoreTestCase(TestCase):

    def setUp(self):
        self.metric = OverlappingScore()

    def test_evaluate(self):
        test_cases = [
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["4", "5", "6"],
                    "wrong_penalty_rate": 2.0,
                },
                -1.0,
            ),
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["4", "DONT_CARE", "6"],
                    "wrong_penalty_rate": 2.0,
                },
                -0.666666666667,
            ),
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["4", "2", "6"],
                    "wrong_penalty_rate": 2.0,
                },
                -0.33333333333333,
            ),
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["DONT_CARE", "DONT_CARE", "DONT_CARE"],
                    "wrong_penalty_rate": 2.0,
                },
                0.0,
            ),
            (
                {
                    "y_pred": ["1", "DONT_CARE", "DONT_CARE"],
                    "y_true": ["DONT_CARE", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                0.0,
            ),
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["DONT_CARE", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                0.6666666666666667,
            ),
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["5", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                0.3333333333333333,
            ),
            (
                {
                    "y_pred": ["DONT_CARE", "DONT_CARE", "DONT_CARE"],
                    "y_true": ["DONT_CARE", "DONT_CARE", "DONT_CARE"],
                    "wrong_penalty_rate": 2.0,
                },
                1.0,
            ),
            (
                {
                    "y_pred": ["1", "2", "3"],
                    "y_true": ["1", "2", "3"],
                    "wrong_penalty_rate": 2.0,
                },
                1.0,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = self.metric.evaluate(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
