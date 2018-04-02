from unittest import TestCase
from ..intent_accuracy_score_with_threshold import (
    intent_accuracy_score_with_threshold,
)


class AccuracyScorewithThresholdTestCase(TestCase):

    def test_intent_accuracy_score_with_threshold_default(self):
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
                        ],
                    ],
                    "y_trues": ["1", "3"],
                },
                1. / 2,
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
                    ],
                    "y_trues": ["1", "2"],
                },
                1. / 2,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_accuracy_score_with_threshold(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_intent_accuracy_score_with_threshold_change_theshold(self):
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
                        ],
                    ],
                    "y_trues": ["1", "3"],
                    "threshold": 0.8,
                },
                0.0,
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
                    ],
                    "y_trues": ["1", "2"],
                    "threshold": 0.2,
                },
                1.0,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_accuracy_score_with_threshold(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_intent_accuracy_score_with_threshold_change_normalize_false(self):
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
                        ],
                    ],
                    "y_trues": ["1", "3"],
                    "normalize": False,
                },
                1,
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
                    ],
                    "y_trues": ["1", "2"],
                    "normalize": False,
                },
                1,
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
                            {"intent": "2", "score": 0.8},
                        ],
                    ],
                    "y_trues": ["1", "3"],
                    "threshold": 0.8,
                    "normalize": False,
                },
                0,
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
                    ],
                    "y_trues": ["1", "2"],
                    "threshold": 0.2,
                    "normalize": False,
                },
                2,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_accuracy_score_with_threshold(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)

    def test_intent_accuracy_score_with_threshold_change_normalize_sample_weight(self):
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
                        ],
                    ],
                    "y_trues": ["1", "3"],
                    "sample_weight": [0.3, 0.5],
                },
                0.3 / (0.3 + 0.5),
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
                    ],
                    "y_trues": ["1", "2"],
                    "sample_weight": [0.3, 0.5],
                },
                0.3 / (0.3 + 0.5),
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
                            {"intent": "2", "score": 0.8},
                        ],
                    ],
                    "y_trues": ["1", "3"],
                    "normalize": False,
                    "sample_weight": [0.3, 0.5],
                },
                0.3,
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
                    ],
                    "y_trues": ["1", "2"],
                    "threshold": 0.2,
                    "normalize": False,
                    "sample_weight": [0.3, 0.5],
                },
                0.8,
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = intent_accuracy_score_with_threshold(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
