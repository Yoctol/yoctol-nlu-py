from typing import List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from .utils import preprocess_entity_prediction
from ynlu.sdk.evaluation import NOT_ENTITY


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
