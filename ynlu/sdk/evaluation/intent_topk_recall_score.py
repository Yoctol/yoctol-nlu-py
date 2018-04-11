from typing import List, Dict


def single__intent_topk_recall_score(
        intent_prediction: List[Dict[str, str]],
        y_true: List[str],
        k: int = 1,
    ) -> float:
    """Compute the Recall of a single utterance with multi-intents

    Recall of a single utterance is defined as the proportion of correctly
    predicted labels to the total number of the predicted label.
    It can be formulated as

    .. math::
      \\text{Recall of single utterance}=\\frac{|\\text{pred}_i \\cap \\text{true}_i|}{|\\text{pred}_i|}

    Args:
        intent_prediction (a list of dictionaries):
            A sorted intent prediction (by score) of a single utterance.
        y_true (a list of strings):
            The corresponding true intent of that utterance.
            Note that it can be more than one intents.
        k (an integer):
            The top k prediction of intents we take for computing Recall.

    Returns:
        recall score (a float):
             Recall of a single utterance given top k prediction.

    Examples:
        >>> intent_prediction, _ = model.predict("I like apple.")
        >>> print(intent_prediction)
        [
            {"intent": "blabla", "score": 0.7},
            {"intent": "ohoh", "score": 0.2},
            {"intent": "preference", "score": 0.1},
        ]
        >>> recall = single__intent_topk_recall_score(
            intent_prediction=intent_prediction,
            y_true=["preference", "ohoh"],
            k=2,
        )
        >>> print(recall)
        0.5
    """ # noqa
    top_k_pred = [pred["intent"] for pred in intent_prediction[: k]]
    recall_score = (
        len(set(y_true) & set(top_k_pred)) /
        len(top_k_pred)
    )
    return recall_score


def intent_topk_recall_score(
        intent_predictions: List[List[Dict[str, str]]],
        y_trues: List[List[str]],
        k: int = 1,
    ) -> float:
    """Compute the Recall of all utterances with multi-intents

    Please take a look at function **single__intent_topk_recall_score** first.
    This function is JUST a batch version of that. It would send all data to
    that function, then collect and average the output.

    .. math::
      \\text{Recall of all utterances}=\\frac{1}{n}\\sum_{i=1}^{n}\\frac{|\\text{pred}_i \\cap \\text{true}_i|}{|\\text{pred}_i|}

    """  # noqa
    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent predictions and labels must have same amount!!!",
        )
    recall_scores = []
    for y_pred, y_true in zip(intent_predictions, y_trues):
        recall_scores.append(
            single__intent_topk_recall_score(
                intent_prediction=y_pred,
                y_true=y_true,
                k=k,
            ),
        )
    return sum(recall_scores) / len(intent_predictions)
