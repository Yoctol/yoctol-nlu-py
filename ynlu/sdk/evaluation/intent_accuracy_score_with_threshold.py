from typing import List, Dict
from sklearn.metrics import accuracy_score

from .utils import preprocess_intent_prediction_by_threshold


def intent_accuracy_score_with_threshold(
        intent_predictions: List[List[Dict[str, str]]],
        y_trues: List[str],
        threshold: float = 0.5,
        normalize: bool = True,
        sample_weight: List[float] = None,
    ) -> float:
    """Top1 accuracy classification score subject to score > threshold

    Only the top1 predicted intent by score will be considered
    when computing accuracy. Moreover, the score of the predicted
    label should be more than threshold. Otherwise, it would be
    replaced with a ``UNK`` token before computing accuracy score.
    That is to say, the situation ``correctly classified`` occurs
    only when two requirements are satisfied:
        1. the top1 predicted label is the same as true label and
        2. score of predicted label is larger than threshold.


    Args:
        intent_predictions (list of list of dicts):
            A list of intent_prediction which can contains all possible
            intent sorted by score.
        y_trues (list of strings): A list of ground truth (correct) intents.
        threshold (float):
            A threshold which limits the efficacy of top1
            predicted intent if its score is less than threshold.
        normalize (bool):
            If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
        sample_weight (list of float):
            Sample weights.

    Returns:
        float:
            The fraction of correctly classified samples, if
            ``normalize == True``. Otherwise, the number of correctly
            classified sample.

    Examples:
        >>> from ynlu.sdk.evaluation import intent_accuracy_score_with_threshold
        >>> intent_accuracy_score_with_threshold(
                intent_predictions=[
                    [{"intent": "a", "score": 0.7}],
                    [{"intent": "b", "score": 0.3}],
                ],
                y_trues=["a", "b"],
            )
        >>> 0.5

    """

    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent predictions and labels must have same amount!!!",
        )
    y_preds = preprocess_intent_prediction_by_threshold(
        intent_predictions=intent_predictions,
        threshold=threshold,
    )
    return accuracy_score(
        y_true=y_trues,
        y_pred=y_preds,
        normalize=normalize,
        sample_weight=sample_weight,
    )
