from typing import List

from .utils import preprocess_entity_prediction
from ynlu.sdk.evaluation import NOT_ENTITY


def single__entity_overlapping_score(
        utterance: str,
        entity_prediction: List[dict],
        y_true: List[str],
        wrong_penalty_rate: float = 2.0,
    ) -> float:
    """Overlapping Score of a Single Utterance

    Examine the true and predicted entities in character-level. Then,
    compute a score to represent the overlapping rate between them.

    Args:
        utterance (a string):
            The input when calling `model.predict()`.
        entity_prediction (a list of dictionaries):
            Entity part of output when callng `model.predict()` with the above
            utterance as input.
        y_true (a list of strings):
            True entity list of that utterance.
        wrong_penalty_rate (float, default is 2.0):
            A penalty score would be given when predicting wrong.

    Returns:
        overlapping score (float):
            If wrong penalty rate is 2.0, then an overlapping score would be ranging
            from -1 to 1. A score -1 represents all entities in an utterance are
            mismatched. On the other hand, score 1 means entities in an utterance
            are perfectly matched.


    Examples:
        >>> from ynlu.sdk.evaluation import single__entity_overlapping_score
        >>> overlapping_score = single__entity_overlapping_score(
                utterance="I like apple.",
                entity_prediction=[
                    {"entity": "DONT_CARE", "value": "I like ", "score": 0.9},
                    {"entity": "fruit", "value": "apple", "score": 0.8},
                    {"entity": "drink", "value": ".", "score": 0.3},
                ],
                y_true=[
                    "DONT_CARE", "DONT_CARE", "DONT_CARE", "DONT_CARE",
                    "DONT_CARE", "DONT_CARE", "DONT_CARE", "fruit",
                    "fruit", "fruit", "fruit", "fruit", "DONT_CARE",
                ],
            )
        >>> print(overlapping_score)
        12 / 13
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

    penalty = 0.0
    for token_p, token_t in zip(y_pred, y_true):
        if token_p == token_t:
            penalty += 0.0
        elif (token_p in [NOT_ENTITY, NOT_ENTITY + "*"]) or (
                token_t in [NOT_ENTITY, NOT_ENTITY + "*"]):
            penalty += 1.0
        else:
            penalty += wrong_penalty_rate
    overlapping_score = 1 - penalty / len(y_pred)
    return overlapping_score


def entity_overlapping_score(
        utterances: List[str],
        entity_predictions: List[List[dict]],
        y_trues: List[List[str]],
        wrong_penalty_rate: float = 2.0,
    ) -> float:
    """Averaged Overlapping Score of all Utterances

    Please take a look at function **single__entity_overlapping_score** first.
    This function is JUST a batch version of that. It would send all data to
    that function, then collect and average the output.

    """
    if len(entity_predictions) != len(y_trues):
        raise ValueError(
            "Entity predictions and labels must have same amount!!!",
        )
    overlapping_scores = []
    for utt, pred, true in zip(utterances, entity_predictions, y_trues):
        overlapping_scores.append(
            single__entity_overlapping_score(
                utterance=utt,
                entity_prediction=pred,
                y_true=true,
                wrong_penalty_rate=wrong_penalty_rate,
            ),
        )
    return sum(overlapping_scores) / len(entity_predictions)
