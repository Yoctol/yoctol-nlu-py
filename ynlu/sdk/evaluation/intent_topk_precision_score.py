from typing import List, Dict


def single__intent_topk_precision_score(
        intent_prediction: List[Dict[str, str]],
        y_true: List[str],
        k: int = 1,
    ) -> float:
    top_k_pred = [pred["intent"] for pred in intent_prediction[: k]]
    precision_score = (
        len(set(y_true) & set(top_k_pred)) /
        len(y_true)
    )
    return precision_score


def intent_topk_precision_score(
        intent_predictions: List[List[Dict[str, str]]],
        y_trues: List[List[str]],
        k: int = 1,
    ) -> float:
    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent prediction ands labels must have same amount!!!",
        )
    precision_scores = []
    for y_pred, y_true in zip(intent_predictions, y_trues):
        precision_scores.append(
            single__intent_topk_precision_score(
                intent_prediction=y_pred,
                y_true=y_true,
                k=k,
            ),
        )
    return sum(precision_scores) / len(intent_predictions)
