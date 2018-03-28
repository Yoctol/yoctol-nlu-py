from typing import List, Dict


def intent_topk_accuracy_score(
        intent_prediction: List[Dict[str, str]],
        y_true: List[str],
        k: int = 1,
    ) -> float:
    top_k_pred = [pred["name"] for pred in intent_prediction[: k]]
    accuracy_score = (
        len(set(y_true) & set(top_k_pred)) /
        len(set(y_true) | set(top_k_pred))
    )
    return accuracy_score


def intent_topk_accuracy_score_overall(
        intent_predictions: List[List[Dict[str, str]]],
        y_trues: List[List[str]],
        k: int=1,
    ) -> float:
    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent prediction and label must have same amount!!!",
        )
    accuracy_scores = []
    for y_pred, y_true in zip(intent_predictions, y_trues):
        accuracy_scores.append(
            intent_topk_accuracy_score(
                intent_prediction=y_pred,
                y_true=y_true,
                k=k,
            ),
        )
    return sum(accuracy_scores) / len(intent_predictions)
