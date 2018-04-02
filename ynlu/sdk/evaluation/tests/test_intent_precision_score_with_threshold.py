from unittest import TestCase
from ..intent_precision_score_with_threshold import (
    intent_precision_score_with_threshold,
)


class PrecisionScorewithThresholdTestCase(TestCase):

    def test_intent_precision_score_with_threshold_default(self):
        test_cases = [
            (
                {
                    "intent_predictions": [
                        [
                            {"intent": "1", "score": 0.7},
                            {"intent": "2"},
                            {"intent": "3"},
                        ],
                        [
                            {"intent": "2", "score": 0.8},
                            {"intent": "4", "score": 0.1},
                        ],
                        [
                            {"intent": "1", "score": 0.8},
                        ],
                    ],
                    "y_trues": ["1", "1", "2"],
                },
                (2 * (1. / 2) + 0.0) / 3,
            ),
            (
                {
                    "intent_predictions": [
                        [
                            {"intent": "1", "score": 0.7},
                            {"intent": "2"},
                            {"intent": "3"},
                        ],
                        [
                            {"intent": "2", "score": 0.3},
                        ],
                        [
                            {"intent": "2", "score": 0.8},
                        ],
                    ],
                    "y_trues": ["1", "2", "1"],
                },
                ((2 * 1) + 0.0) / 3,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_precision_score_with_threshold(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_intent_precision_score_with_threshold_change_theshold(self):
        test_cases = [
            (
                {
                    "intent_predictions": [
                        [
                            {"intent": "1", "score": 0.6},
                            {"intent": "2"},
                            {"intent": "3"},
                        ],
                        [
                            {"intent": "1", "score": 0.8},
                        ],
                        [
                            {"intent": "3", "score": 0.8},
                        ],
                    ],
                    "y_trues": ["1", "3", "3"],
                    "threshold": 0.7,
                },
                (2 * 1.0) / 3,
            ),
            (
                {
                    "intent_predictions": [
                        [
                            {"intent": "1", "score": 0.7},
                            {"intent": "2"},
                            {"intent": "3"},
                        ],
                        [
                            {"intent": "2", "score": 0.3},
                        ],
                        [
                            {"intent": "2", "score": 0.3},
                        ],
                    ],
                    "y_trues": ["1", "2", "1"],
                    "threshold": 0.2,
                },
                ((2 * 1.0) + (1.0 / 2)) / 3,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_precision_score_with_threshold(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
