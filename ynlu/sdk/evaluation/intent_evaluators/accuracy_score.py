from typing import List, Dict
from os.path import dirname

from mkdir_p import mkdir_p
import pandas as pd


class AccuracyScore(object):

    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    def preprocess_prediction(self, prediction: List[Dict[str, str]]):
        candidate_predictions = []
        for i in range(self.top_k):
            candidate_predictions.append(prediction[i]["intent"])
        return candidate_predictions

    def preprocess_label(self, label: str):
        if isinstance(label, str):
            return [label]
        return list(label)

    def evaluate(
            self,
            y_pred: List[str],
            y_true: List[str],
        ) -> bool:
        return set(y_true) <= set(y_pred)

    def run(
            self,
            utterances: List[str],
            predictions: List[Dict[str, str]],
            labels: List[List[str]],
            output_path: str,
        ) -> None:
        result_collection = []
        for prediction, label in zip(predictions, labels):
            p_label = self.preprocess_label(label=label)
            p_pred = self.preprocess_prediction(prediction=prediction)
            result_collection.append(
                (
                    p_label,
                    p_pred,
                    self.evaluate(y_pred=p_pred, y_true=p_label),
                ),
            )
        result_collection = list(zip(*result_collection))
        self.save(
            output_path=output_path,
            utterances=utterances,
            predictions=result_collection[1],
            labels=result_collection[0],
            eval_results=result_collection[2],
        )

    def gen_report(
            self,
            utterances: List[str],
            predictions: List[List[str]],
            labels: List[List[str]],
            eval_results: List[bool],
        ) -> pd.DataFrame:
        report_df = pd.DataFrame()
        report_df["utterance"] = utterances
        report_df["top_{}_result".format(self.top_k)] = eval_results
        report_df["prediction"] = predictions
        report_df["label"] = labels
        return report_df

    def describe(self):
        """
        Give some description about the evaluator
        """
        raise NotImplementedError

    def save(
            self,
            output_path: str,
            utterances: List[str],
            predictions: List[List[str]],
            labels: List[List[str]],
            eval_results: List[bool],
        ):
        mkdir_p(dirname(output_path))
        report_df = self.gen_report(
            utterances=utterances,
            predictions=predictions,
            labels=labels,
            eval_results=eval_results,
        )
        report_df.to_csv(output_path, index=False)
