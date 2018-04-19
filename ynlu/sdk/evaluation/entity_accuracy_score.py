from typing import List

from .utils import preprocess_entity_prediction
from ynlu.sdk.evaluation import NOT_ENTITY


def single__entity_accuracy_score(
        utterance: str,
        entity_prediction: List[dict],
        y_true: List[str],
    ) -> float:
    """Top1 accuracy of entity extraction.

    Examine the true and predicted entities in character-level. Then,
    compute a score to represent the accuracy between them.

    Args:
        utterance (a string):
            The input when calling `model.predict()`.
        entity_prediction (a list of dictionaries):
            Entity part of output when callng `model.predict()` with the above
            utterance as input.
        y_true (a list of strings):
            True entity list of that utterance.

    Returns:
        accuracy score (float):
            Accuracy score ranges from 0 to 1. Score 1 stands for perfectly matching,
            and score 0 means that targets in utterance are all mismatched.


    Examples:
        >>> from ynlu.sdk.evaluation import single__entity_accuracy_score
        >>> accuracy_score = single__entity_accuracy_score(
                utterance="I like apple.",
                entity_prediction=[
                    {"entity": "DONT_CARE", "value": "I like ", "score": 0.9},
                    {"entity": "fruit", "value": "app", "score": 0.8},
                    {"entity": "drink", "value": "le.", "score": 0.3},
                ],
                y_true=[
                    "DONT_CARE", "DONT_CARE", "DONT_CARE", "DONT_CARE",
                    "DONT_CARE", "DONT_CARE", "DONT_CARE", "fruit",
                    "fruit", "fruit", "fruit", "fruit", "DONT_CARE",
                ],
            )
        >>> print(accuracy_score)
        10 / 13
    """
    y_pred = preprocess_entity_prediction(
        utterance=utterance,
        entity_prediction=entity_prediction,
        not_entity=NOT_ENTITY,
    )

    if len(y_pred) != len(y_true):
        raise ValueError(
            "Entity prediction and label must have same length!!!",
        )

    matched = [pred for pred, true in zip(y_pred, y_true) if pred == true]
    accuracy_score = len(matched) / len(y_pred)
    return accuracy_score


def entity_accuracy_score(
        utterances: List[str],
        entity_predictions: List[List[dict]],
        y_trues: List[List[str]],
    ) -> float:
    """Averaged Accuracy Score of all Utterances

    Please take a look at function **single__entity_accuracy_score** first.
    This function is JUST a batch version of that. It would send all data to
    that function, then collect and average the output.

    """
    if len(entity_predictions) != len(y_trues):
        raise ValueError(
            "Entity predictions and labels must have same amount!!!",
        )
    accuracy_scores = []
    for utt, pred, true in zip(utterances, entity_predictions, y_trues):
        accuracy_scores.append(
            single__entity_accuracy_score(
                utterance=utt,
                entity_prediction=pred,
                y_true=true,
            ),
        )
    return sum(accuracy_scores) / len(accuracy_scores)
