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
            {
                "data": [
                    {
                        "x": [1, 2, 3],
                        "y": [4, 5, 6],
                    },
                ],
                "block": False,
                "x_axis_name": "x軸",
                "y_axis_name": "y軸",
                "title": "我是標題",
            },
            {
                "data": [
                    {
                        "x": [1, 2, 3],
                        "y": [4, 5, 6],
                        "label": "第一條",
                    },
                    {
                        "x": [1, 2, 3],
                        "y": [-4, -5, -6],
                        "label": "第二條",
                    },
                ],
                "block": False,
            },
        ]
        for test_case in test_cases:
            plot_lines(**test_case)
