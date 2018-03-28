from typing import List, Dict


def single__intent_topk_recall_score(
        intent_prediction: List[Dict[str, str]],
        y_true: List[str],
        k: int = 1,
    ) -> float:
    top_k_pred = [pred["entity"] for pred in intent_prediction[: k]]
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
