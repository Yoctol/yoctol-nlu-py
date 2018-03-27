from unittest import TestCase
from ..intent_topk_precision_score import (
    intent_topk_precision_score,
    intent_topk_precision_score_overall,
)


class TopKAccuracyScoreTestCase(TestCase):

    def test_top1_precision_score(self):
        test_cases = [
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["1", "3"],
                    "k": 1,
                },
                0.499999999999,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["1"],
                    "k": 1,
                },
                0.999999999999,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["1", "3", "7", "5"],
                    "k": 1,
                },
                0.25,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["5"],
                    "k": 1,
                },
                0.0,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_precision_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_top3_precision_score(self):
        test_cases = [
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["1", "3"],
                    "k": 3,
                },
                1.0,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["1"],
                    "k": 3,
                },
                1.0,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["1", "3", "7", "5"],
                    "k": 3,
                },
                0.5,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                        {"name": "3"},
                    ],
                    "y_true": ["5"],
                    "k": 3,
                },
                0.0,
            ),
            (
                {
                    "intent_prediction": [
                        {"name": "1"},
                        {"name": "2"},
                    ],
                    "y_true": ["1", "3", "7", "5"],
                    "k": 3,
                },
                0.25,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_precision_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_top1_precision_score_overall(self):
        test_cases = [
            (
                {
                    "intent_predictions": [
                        [
                            {"name": "1"},
                            {"name": "2"},
                            {"name": "3"},
                        ],
                        [
                            {"name": "2"},
                            {"name": "4"},
                            {"name": "6"},
                        ],
                        [
                            {"name": "7"},
                            {"name": "3"},
                            {"name": "5"},
                        ],
                    ],
                    "y_trues": [["1"], ["3"], ["5"]],
                    "k": 1,
                },
                0.33333333333,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_precision_score_overall(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_top3_precision_score_overall(self):
        test_cases = [
            (
                {
                    "intent_predictions": [
                        [
                            {"name": "1"},
                            {"name": "2"},
                        ],
                        [
                            {"name": "2"},
                            {"name": "4"},
                            {"name": "6"},
                        ],
                        [
                            {"name": "7"},
                        ],
                    ],
                    "y_trues": [["1"], ["3"], ["7"]],
                    "k": 3,
                },
                0.6666666666666,
            ),
            (
                {
                    "intent_predictions": [
                        [
                            {"name": "1"},
                            {"name": "2"},
                        ],
                        [
                            {"name": "2"},
                            {"name": "4"},
                            {"name": "6"},
                        ],
                        [
                            {"name": "7"},
                        ],
                    ],
                    "y_trues": [["1", "10", "12"], ["3", "2"], ["7", "9"]],
                    "k": 3,
                },
                0.4444444444444444,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_precision_score_overall(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
