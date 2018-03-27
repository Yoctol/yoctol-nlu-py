from unittest import TestCase
from ..intent_topk_accuracy_score import intent_topk_accuracy_score


class TopKAccuracyScoreTestCase(TestCase):

    def test_top1_accuracy_score(self):
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
                result = intent_topk_accuracy_score(**test_case[0])
                self.assertAlmostEqual(test_case[1], result)
