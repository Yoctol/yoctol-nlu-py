from unittest import TestCase
from ..utils import (
    remove_annotation,
    preprocess_annotated_utterance,
    preprocess_entity_prediction,
    preprocess_intent_prediction_by_threshold,
)


class OverlappingScoreTestCase(TestCase):

    def test_remove_annotation(self):
        test_cases = [
            (
                "><abc<a>eee</a>123<b>46hhh</b>890",
                "><abceee12346hhh890",
            ),
            (
                "<a>123<b>456</b>",
                "<a>123456",
            ),
            (
                "<<a>>123<b>",
                "<<a>>123<b>",
            ),
            (
                "<a></a><b><c><d><e><f>",
                "<b><c><d><e><f>",
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = remove_annotation(test_case[0])
                self.assertEqual(test_case[1], result)

    def test_preprocess_annotated_utterance(self):
        test_cases = [
            (
                "><<a>eee</a>123<b>46hhh</b>890",
                ["n", "n", "a", "a", "a", "n", "n", "n",
                 "b", "b", "b", "b", "b", "n", "n", "n"],
            ),
            (
                "<a>eee</a>1>><<23<b>46hhh</b>89><0><",
                ["a", "a", "a", "n", "n", "n", "n", "n", "n", "n",
                 "b", "b", "b", "b", "b", "n", "n", "n", "n", "n", "n", "n"],
            ),
            (
                "<a>123<b>456</b>",
                ["n", "n", "n", "n", "n", "n", "b", "b", "b"],
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = preprocess_annotated_utterance(
                    annotated_utterance=test_case[0],
                    not_entity="n",
                )
                self.assertEqual(test_case[1], result)

    def test_preprocess_entity_prediction(self):
        test_cases = [
            (
                "++a-b-c",
                [
                    {
                        "entity": "1",
                        "value": "a",
                    },
                    {
                        "entity": "2",
                        "value": "b",
                    },
                    {
                        "entity": "3",
                        "value": "c",
                    },
                ],
                ["n", "n", "1", "n", "2", "n", "3"],
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = preprocess_entity_prediction(
                    utterance=test_case[0],
                    entity_prediction=test_case[1],
                    not_entity="n",
                )
                self.assertEqual(test_case[2], result)

    def test_preprocess_intent_prediction_by_threshold(self):
        test_cases = [
            (
                [[{"intent": "a", "score": 0.7}]],
                ["a"],
            ),
            (
                [[{"intent": "a", "score": 0.3}]],
                ["UNK"],
            ),
            (
                [
                    [
                        {"intent": "a", "score": 0.3},
                    ],
                    [
                        {"intent": "b", "score": 0.7},
                        {"intent": "c", "score": 0.2},
                    ],
                ],
                ["UNK", "b"],
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = preprocess_intent_prediction_by_threshold(
                    test_case[0],
                )
                self.assertEqual(test_case[1], result)
