import itertools
from typing import List, Tuple
from os.path import join, dirname, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


ROOT_DIR = dirname(dirname(abspath(__file__)))
FONT_PATH = join(ROOT_DIR, "data/simhei.ttf")
DEFAULT_FONT = fm.FontProperties(fname=FONT_PATH)


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
        font: fm.FontProperties = DEFAULT_FONT,
        block: bool = True,
    ) -> None:

    _check_input_data_format(
        confusion_matrix=confusion_matrix,
        indices=indices,
    )

    plt.figure(figsize=figure_size)

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
    plt.title(title, fontproperties=font)
    plt.colorbar()
    tick_marks = np.arange(len(indices))
    plt.xticks(tick_marks, indices, rotation=45, fontproperties=font)
    plt.yticks(tick_marks, indices, fontproperties=font)

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

    if output_path is not None:
        plt.savefig(output_path)
        print("Saving confusion matrix figure to {}".format(output_path))
    else:
        plt.show(block=block)
