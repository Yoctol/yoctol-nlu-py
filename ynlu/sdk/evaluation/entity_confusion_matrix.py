import itertools
from typing import List, Tuple
from os.path import abspath, join, dirname

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix


from .utils import preprocess_entity_prediction
from ynlu.sdk.evaluation import NOT_ENTITY


ROOT_DIR = dirname(abspath(__file__))
FONT_PATH = join(ROOT_DIR, "data/simhei.ttf")
DEFAULT_FONT = fm.FontProperties(fname=FONT_PATH)


def entity_confusion_matrix(
        utterances: List[str],
        entity_predictions: List[List[dict]],
        y_trues: List[List[str]],
    ) -> Tuple[np.ndarray, List[str]]:
    flatten_y_pred = []
    flatten_y_true = []
    for utt, pred, y_true in zip(utterances, entity_predictions, y_trues):
        y_pred = preprocess_entity_prediction(
            utterance=utt,
            entity_prediction=pred,
            not_entity=NOT_ENTITY,
        )
        if len(y_pred) != len(y_true):
            raise ValueError(
                "Entity prediction and label must have same length!!!",
            )
        flatten_y_pred.extend(y_pred)
        flatten_y_true.extend(y_true)

    unique_entities = sorted(
        list(
            set(flatten_y_pred) |
            set(flatten_y_true),
        ),
    )
    return confusion_matrix(
        y_true=flatten_y_true,
        y_pred=flatten_y_pred,
        labels=unique_entities,
    ), unique_entities


def entity_confusion_matrix_figure(
        confusion_matrix,
        unique_entities: List[str],
        normalize: bool = False,
        title: str = "Confusion matrix for entity",
        figure_size: Tuple[int, int] = (8, 6),
        cmap=plt.cm.Blues,
        font=DEFAULT_FONT,
    ) -> None:

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
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(unique_entities))
    plt.xticks(tick_marks, unique_entities, rotation=45, fontproperties=font)
    plt.yticks(tick_marks, unique_entities, fontproperties=font)

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
    plt.show()
