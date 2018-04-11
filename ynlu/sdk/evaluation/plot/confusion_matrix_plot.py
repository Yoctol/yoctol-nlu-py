import itertools
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import plt_set_font_style


def _check_input_data_format(
        confusion_matrix: np.ndarray,
        indices: List[str],
    ) -> None:
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            "confusion matrix should be a squared matrix.",
        )
    if confusion_matrix.shape[0] != len(indices):
        raise ValueError(
            """
            The shape of confusion matrix should be the same as
            the length of indices.
            """,
        )


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        indices: List[str],
        normalize: bool = False,
        output_path: str = None,
        title: str = "Confusion Matrix",
        figure_size: Tuple[int, int] = (8, 6),
        cmap: plt.cm = plt.cm.Blues,
        font_style_path: str = None,
        dpi: int = 300,
        block: bool = True,
    ) -> None:
    """Plot confusion matrix

    Args:
        confusion_matrix (a squared numpy matrix):
            The outcome of ``sklearn.metrics.confusion_matrix`` is expected.
        indices (list of string):
            List of strings to index the matrix.
            The expected indices should be the same as the input argument ``labels``
            when you generate confusion matrix.
        normalize (bool):
            If True,

        .. math::
          \\text{confusion matrix}_{ij} = \\frac{\\text{confusion matrix}_{ij}}{\sum_{j=1}^{n}\\text{confusion matrix}_{ij}}

        output_path (string, default = None):
            The place where the output figure would be stored.
            If it is None, the figure will be shown on screen
            automatically.
        title (string, default = "Confusion Matrix"):
            The title of the figure.
        figure_size (a pair of integers, default = (8, 6)):
            The height and width of the output figure.
        cmap (color map):
            Matplotlib built-in colormaps.
        font_style_path (path of font style):
            If None, ``simhei.ttf`` will be used as default font style.
            Chinese characters are supported in this font style.
        dpi (int, default = 300):
            The resolution in dots per inch.
        block (bool):
            If False, the figure will not be shown up even if output_path
            is None. This argument is left for unittest.

    Returns: None

    Example:
        >>> import numpy as np
        >>> from ynlu.sdk.evaluation.plot import plot_confusion_matrix
        >>> plot_confusion_matrix(
                confusion_matrix=np.random.rand(4,4),
                indices=["a", "b", "c", "d"],
            )

    """ # noqa
    _check_input_data_format(
        confusion_matrix=confusion_matrix,
        indices=indices,
    )

    plt.figure(figsize=figure_size, dpi=dpi)

    if normalize:
        confusion_matrix = (
            confusion_matrix.astype(float) /
            confusion_matrix.sum(axis=1)[:, np.newaxis]
        )

    plt.imshow(
        confusion_matrix,
        interpolation='nearest',
        cmap=cmap,
    )
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(indices))
    plt.xticks(tick_marks, indices, rotation=45)
    plt.yticks(tick_marks, indices)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(
        range(confusion_matrix.shape[0]),
        range(confusion_matrix.shape[1]),
    ):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt_set_font_style(font_style_path=font_style_path)

    if output_path is not None:
        plt.savefig(output_path)
        print("Saving confusion matrix figure to {}".format(output_path))
    else:
        plt.show(block=block)
