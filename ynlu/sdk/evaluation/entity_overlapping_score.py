from typing import List

from .utils import preprocess_entity_prediction
from ynlu.sdk.evaluation import NOT_ENTITY


def single__entity_overlapping_score(
        utterance: str,
        entity_prediction: List[dict],
        y_true: List[str],
        wrong_penalty_rate: float = 2.0,
    ) -> float:

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
