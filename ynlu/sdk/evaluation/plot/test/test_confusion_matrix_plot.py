from unittest import TestCase

import numpy as np

from ..confusion_matrix_plot import (
    plot_confusion_matrix,
)


class ConfusionMatrixTestCase(TestCase):

    def test_plot_confusion_matrix(self):
        test_cases = [
            {
                "confusion_matrix": np.array(
                    [[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]),
                "indices": ["a", "b", "c"],
                "title": "test",
                "block": False,
            },
            {
                "confusion_matrix": np.array(
                    [[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]),
                "indices": ["a", "b", "c"],
                "normalize": True,
                "title": "test",
                "block": False,
            },
            {
                "confusion_matrix": np.array(
                    [[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]),
                "indices": ["第一", "第二", "第三"],
                "title": "這是個標題",
                "block": False,
            },
        ]
        for test_case in test_cases:
            plot_confusion_matrix(**test_case)
