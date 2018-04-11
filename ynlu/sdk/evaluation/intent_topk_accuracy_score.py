from typing import List, Dict


def single__intent_topk_accuracy_score(
        intent_prediction: List[Dict[str, str]],
        y_true: List[str],
        k: int = 1,
    ) -> float:
    """Compute the Accuracy of a single utterance with multi-intents

    Accuracy of a single utterance is defined as the proportion of
    correctly predicted labels to the total number (predicted and true)
    of labels. It can be formulated as

    .. math::
      \\text{Accuracy of single utterance}=\\frac{|\\text{pred}_i \\cap \\text{true}_i|}{|\\text{true}_i \\cap \\text{pred}_i|}

    Args:
        intent_prediction (a list of dictionaries):
            A sorted intent prediction (by score) of a single utterance.
        y_true (a list of strings):
            The corresponding true intent of that utterance.
            Note that it can be more than one intents.
        k (an integer):
            The top k prediction of intents we take for computing accuracy.

    Returns:
        accuracy score (a float):
             accuracy of a single utterance given top k prediction.

    Examples:
        >>> intent_prediction, _ = model.predict("I like apple.")
        >>> print(intent_prediction)
        [
            {"intent": "blabla", "score": 0.7},
            {"intent": "ohoh", "score": 0.2},
            {"intent": "preference", "score": 0.1},
        ]
        >>> accuracy = single__intent_topk_accuracy_score(
            intent_prediction=intent_prediction,
            y_true=["preference", "ohoh", "YY"],
            k=2,
        )
        >>> print(accuracy)
        0.2499999
    """ # noqa
    top_k_pred = [pred["intent"] for pred in intent_prediction[: k]]
    accuracy_score = (
        len(set(y_true) & set(top_k_pred)) /
        len(set(y_true) | set(top_k_pred))
    )
    return accuracy_score


def intent_topk_accuracy_score(
        intent_predictions: List[List[Dict[str, str]]],
        y_trues: List[List[str]],
        k: int=1,
    ) -> float:
    """Compute the Accuracy of all utterances with multi-intents

    Please take a look at function **single__intent_topk_accuracy_score** first.
    This function is JUST a batch version of that. It would send all data to
    that function, then collect and average the output.

    .. math::
      \\text{Accuracy of all utterances}=\\frac{1}{n}\\sum_{i=1}^{n}\\frac{|\\text{pred}_i \\cap \\text{true}_i|}{|\\text{true}_i|}

    """  # noqa
    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent predictions and labels must have same amount!!!",
        )
    accuracy_scores = []
    for y_pred, y_true in zip(intent_predictions, y_trues):
        accuracy_scores.append(
            single__intent_topk_accuracy_score(
                intent_prediction=y_pred,
                y_true=y_true,
                k=k,
            ),
        )
    return sum(accuracy_scores) / len(intent_predictions)
