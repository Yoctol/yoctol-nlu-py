from typing import List, Dict


def intent_topk_accuracy_score(
        intent_predictions: List[Dict[str, str]],
        y_trues: List[List[str]],
        k: int = 1,
    ) -> float:
    if len(intent_predictions) != len(y_trues):
        raise ValueError(
            "Intent prediction and label must have same amount!!!",
        )
    bool_result = []
    for y_pred, y_true in zip(intent_predictions, y_trues):
        top_k_pred = [pred["name"] for pred in y_pred[: k]]
        bool_result.append(
            set(y_true).issubset(set(top_k_pred)),
        )
    return sum(bool_result) / len(intent_predictions)
