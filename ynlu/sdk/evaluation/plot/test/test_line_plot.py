from unittest import TestCase

from ..line_plot import (
    plot_lines,
)


class LinePlotTestCase(TestCase):

    def test_plot_lines(self):
        test_cases = [
            {
                "data": [
                    {
                        "x": [1, 2, 3],
                        "y": [4, 5, 6],
                    },
                ],
                "block": False,
            },
        ]
        for test_case in test_cases:
            plot_lines(**test_case)
