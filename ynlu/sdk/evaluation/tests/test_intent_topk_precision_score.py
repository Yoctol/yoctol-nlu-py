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
                1. / 2,
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
                1. / 1,
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
                1. / 4,
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
                0. / 1,
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
                2. / 2,
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
                1. / 1,
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
                2. / 4,
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
                0. / 1,
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
                1. / 4,
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
                (1. / 1 + 0. / 1 + 0. / 1) / 3,
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
                (1. / 1 + 0. / 1 + 1. / 1) / 3,
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
                (1. / 3 + 1. / 2 + 1. / 2) / 3,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_precision_score_overall(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
