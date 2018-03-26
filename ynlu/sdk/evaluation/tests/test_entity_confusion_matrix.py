from unittest import TestCase

import numpy as np

from ..entity_confusion_matrix import (
    entity_confusion_matrix,
    # entity_confusion_matrix_figure,
)


class EntityConfusionMatrixTestCase(TestCase):

    def test_entity_confusion_matrix(self):
        test_cases = [
            (
                {
                    "entity_predictions": [
                        [
                            {"name": "1", "value": "1"},
                            {"name": "2", "value": "2"},
                            {"name": "3", "value": "3"},
                        ],
                    ],
                    "utterances": ["123"],
                    "y_trues": [["4", "5", "6"]],
                },
                (
                    np.array(
                        [[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]],
                    ),
                    ['1', '2', '3', '4', '5', '6'],
                ),
            ),
        ]
        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i):
                result = entity_confusion_matrix(**test_case[0])
                self.assertEqual(
                    test_case[1][0].tolist(),
                    result[0].tolist(),
                )
                self.assertEqual(
                    test_case[1][1], result[1],
                )

    # def test_entity_confusion_matrix_figure(self):
    #     test_cases = [
    #         {
    #             "confusion_matrix": np.array(
    #                 [[1, 2, 3],
    #                  [4, 5, 6],
    #                  [7, 8, 9]]),
    #             "unique_entities": ["a", "b", "c"],
    #             "title": "test",
    #         },
    #         {
    #             "confusion_matrix": np.array(
    #                 [[1, 2, 3],
    #                  [4, 5, 6],
    #                  [7, 8, 9]]),
    #             "unique_entities": ["a", "b", "c"],
    #             "normalize": True,
    #             "title": "test",
    #         },
    #     ]
    #     for test_case in test_cases:
    #         entity_confusion_matrix_figure(**test_case)
