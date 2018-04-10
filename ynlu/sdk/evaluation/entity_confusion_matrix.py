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
    """Confusion Matrix for Evaluating Entity Predictions

    Preprocess a list of raw entity predictions and
    compute confusion matrix by calling ``sklearn.metrics.confusion_matrix``.

    Args:
        utterances (a list of strings):
            The original input that produce the entity_predictions below
            by calling `model.predict()`.
        entity_predictions (a list of entity_predictions):
            Entity part of output when callng `model.predict()` with the above
            utterance as input.
        y_trues (a list of y_true):
            A list of true entity lists of that utterance.

    Returns:
        confusion_matrix (numpy 2D array):
            By definition a confusion matrix C is such that :math:`C_{i, j}`
            is equal to the number of observations known to be in group i
            but predicted to be in group j. If a confusion matrix is a
            diagonal matrix, we can know that the predictions and true
            labels are perfectly matched.

        unique_entities (list of strings):
            It records the meaning of group :math:`i`, :math:`j`
            of confusion matrix :math:`C_{i, j}`.

    Examples:
        >>> from ynlu.sdj.evaluation import entity_confusion_matrix
        >>> confusion_matrix, unique_entities = entity_confusion_matrix(
                utterances=["I like apple."],
                entity_predictions=[
                    [
                        {"entity": "DONT_CARE", "value": "I like ", "score": 0.9},
                        {"entity": "fruit", "value": "apple", "score": 0.8},
                        {"entity": "drink", "value": ".", "score": 0.3},
                    ],
                ],
                y_trues=[
                    [
                        "DONT_CARE", "DONT_CARE", "DONT_CARE", "DONT_CARE",
                        "DONT_CARE", "DONT_CARE", "DONT_CARE", "fruit",
                        "fruit", "fruit", "fruit", "fruit", "DONT_CARE",
                    ],
                ],
            )
        >>> print(unique_entities)
        ["DONT_CARE", "fruit", "drink"]
        >>> print(confusion_matrix)
        np.array([[7, 0, 1], [0, 5, 0], [0, 0, 0]])

    """
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
