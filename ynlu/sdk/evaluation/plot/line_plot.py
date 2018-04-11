from typing import List, Tuple

import seaborn as sns
import matplotlib.pyplot as plt

from .utils import plt_set_font_style


def _check_data_format(data: List[dict]):
    for i, datum in enumerate(data):
        if "x" not in datum:
            raise KeyError("No.{}: datum {} has no x.".format(i, datum))
        if "y" not in datum:
            raise KeyError("No.{}: datum {} has no y.".format(i, datum))
        if len(datum["x"]) != len(datum["y"]):
            raise KeyError(
                "No.{}: x, y in datum must have the same length.".format(i))


def plot_lines(
        data: List[dict],
        title: str = "figure",
        x_axis_name: str = "x",
        y_axis_name: str = "y",
        figure_size: Tuple[int, int]=(8, 6),
        output_path: str = None,
        color_palette: sns.color_palette = None,
        font_style_path: str = None,
        dpi: int = 300,
        block: bool = True,
    ) -> None:
    """Plot y versus x as lines and/or markers


    Args:
        data (list of dictionaries):
            Dictionaries containing arguments and key words for
            ``matplotlib.pyplot.plot``. Basic arguments are:
            ``x``, ``y``, ``label`` (the name of line).
        title (string, default = "figure"):
            The title of the figure.
        x_axis_name (string, default = "x"):
            The name to be shown on x axis.
        y_axis_name(string, default = "y"):
            The name to be shown on y axis.
        figure_size (a pair of integers, default = (8, 6)):
            The height and width of the output figure.
        output_path (string, default = None):
            The place where the output figure would be stored.
            If it is None, the figure will be shown on screen
            automatically.
        color_palette (seaborn color palette object):
            Please take a look at
            ``https://seaborn.pydata.org/generated/seaborn.color_palette.html``
            for more details.
        font_style_path (path of font style):
            If None, ``simhei.ttf`` will be used as default font style.
            Chinese characters are supported in this font style.
        dpi (int, default = 300):
            The resolution in dots per inch.
        block (bool):
            if False, the figure will not be shown up even if output_path
            is None. This argument is left for unittest.

    Returns: None

    Example:
        >>> from ynlu.sdk.evaluation.plot import plot_lines
        >>> plot_lines(
                data=[
                    {"x": [1, 2, 3], "y": [4, 5, 6], "label": "line1"},
                    {"x": [6, 7, 8], "y": [9, 10, 11], "label": "line2"},
                ],
            )

    """

    _check_data_format(data=data)

    plt.figure(figsize=figure_size, dpi=dpi)

    if color_palette is None:
        color_palette = sns.color_palette("Set2", 10)
    sns.set_palette(color_palette)
    default_plot_params = {
        "linewidth": 2,
        "alpha": 0.7,
        "linestyle": "-",
        "marker": "o",
    }

    lines = []
    line_names = []
    for i, input_datum in enumerate(data):
        plot_params = default_plot_params
        plot_params.update(input_datum)
        del plot_params["x"], plot_params["y"]
        line, = plt.plot(input_datum["x"], input_datum["y"], **plot_params)
        lines.append(line)
        line_names.append(input_datum.get("label", "line_" + str(i)))

    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.legend(
        handles=lines,
        labels=line_names,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.,
    )
    plt.subplots_adjust(right=0.8)
    plt_set_font_style(font_style_path=font_style_path)

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show(block=block)
