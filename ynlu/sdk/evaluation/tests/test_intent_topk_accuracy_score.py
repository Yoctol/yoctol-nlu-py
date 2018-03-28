from unittest import TestCase
from ..intent_topk_accuracy_score import (
    intent_topk_accuracy_score,
    intent_topk_accuracy_score_overall,
)


class TopKAccuracyScoreTestCase(TestCase):

    def test_top1_accuracy_score(self):
        test_cases = [
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["1", "3"],
                    "k": 1,
                },
                0.499999999999,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["1"],
                    "k": 1,
                },
                0.999999999999,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["1", "3", "7", "5"],
                    "k": 1,
                },
                0.25,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["5"],
                    "k": 1,
                },
                0.0,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_accuracy_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_top3_accuracy_score(self):
        test_cases = [
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["1", "3"],
                    "k": 3,
                },
                0.6666666666667,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["1"],
                    "k": 3,
                },
                0.3333333333333,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["1", "3", "7", "5"],
                    "k": 3,
                },
                0.39999999999999,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                        {"entity": "3"},
                    ],
                    "y_true": ["5"],
                    "k": 3,
                },
                0.0,
            ),
            (
                {
                    "intent_prediction": [
                        {"entity": "1"},
                        {"entity": "2"},
                    ],
                    "y_true": ["1", "3", "7", "5"],
                    "k": 3,
                },
                0.19999999999999,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_accuracy_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_top1_accuracy_score_overall(self):
        test_cases = [
            (
                {
                    "intent_predictions": [
                        [
                            {"entity": "1"},
                            {"entity": "2"},
                            {"entity": "3"},
                        ],
                        [
                            {"entity": "2"},
                            {"entity": "4"},
                            {"entity": "6"},
                        ],
                        [
                            {"entity": "7"},
                            {"entity": "3"},
                            {"entity": "5"},
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
                result = intent_topk_accuracy_score_overall(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_top3_accuracy_score_overall(self):
        test_cases = [
            (
                {
                    "intent_predictions": [
                        [
                            {"entity": "1"},
                            {"entity": "2"},
                        ],
                        [
                            {"entity": "2"},
                            {"entity": "4"},
                            {"entity": "6"},
                        ],
                        [
                            {"entity": "7"},
                        ],
                    ],
                    "y_trues": [["1"], ["3"], ["7"]],
                    "k": 3,
                },
                0.49999999999999,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_topk_accuracy_score_overall(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
