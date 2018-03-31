from typing import List, Dict
from sklearn.metrics import precision_score

from .utils import preprocess_intent_prediction_by_threshold
from ynlu.sdk.evaluation import UNKNOWN


def intent_precision_score_with_threshold(
        intent_predictions: List[List[Dict[str, str]]],
        y_trues: List[str],
        threshold: float = 0.5,
        average: str = "weighted",
        sample_weight: List[float] = None,
    ) -> float:
    """Top1 precision classification score subject to score > threshold

    Only the top1 predicted intent by score will be considered
    when computing precision. Moreover, the score of the predicted
    label should be more than threshold. Otherwise, it would be
    replaced with a ``UNK`` token before computing precision score.
    That is to say, the situation ``correctly classified`` occurs
    only when two requirements are satisfied:
        1. the top1 predicted label is the same as true label and
        2. score of predicted label is larger than threshold.


    Args:
        intent_predictions (list of list of dicts):
            A list of intent_prediction which can contains all possible
            intent sorted by score.
        y_trues (list of strings):
            A list of ground truth (correct) intents.
        threshold (float):
            A threshold which limits the efficacy of top1
            predicted intent if its score is less than threshold.
        average (string):
            Options are as follows:
                [‘None’, ‘binary’, ‘micro’, ‘macro’, ‘samples’, ‘weighted’(default)].
            Please look at
             ``http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html``
            for more details.
        sample_weight (list of float):
            Sample weights.

    Returns:
        float (if average is not None):
            Precision of the positive class in binary classification
            or weighted average of the precision of each class for
            the multiclass task.
        or
        array of float (shape = [n_unique_labels]):
            A list of precision of each class.

    Examples:
        >>> from ynlu.sdk.evaluation import intent_precision_score_with_threshold
        >>> intent_precision_score_with_threshold(
                intent_predictions=[
                    [{"intent": "a", "score": 0.7}],
                    [{"intent": "b", "score": 0.3}],
                    [{"intent": "b", "score": 0.8}],
                ],
                y_trues=["a", "b", "a"],
            )
        >>> 0.333333

    """

    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent predictions and labels must have same amount!!!",
        )
    y_preds = preprocess_intent_prediction_by_threshold(
        intent_predictions=intent_predictions,
        threshold=threshold,
    )
    labels = sorted(set(y_trues) | set(y_preds) - set([UNKNOWN]))
    return precision_score(
        y_true=y_trues,
        y_pred=y_preds,
        average=average,
        labels=labels,
        sample_weight=sample_weight,
    )
