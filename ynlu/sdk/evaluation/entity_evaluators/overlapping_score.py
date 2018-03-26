from typing import List
from os.path import join, dirname

import pandas as pd
from bistiming import SimpleTimer
from mkdir_p import mkdir_p

from .base_entity_evaluator import BaseEntityEvaluator


class OverlappingScore(BaseEntityEvaluator):

    def __init__(self):
        self.metric_name = "entity_overlapping_score"

    def run(
            self,
            output_path: str,
            annotated_utterances: List[str],
            predictions: List[List[str]],
            labels: List[List[str]],
            wrong_penalty_rate: float = 2.0,
            verbose: int = 0,
        ):
        mkdir_p(dirname(output_path))
        enriched_metric_name = self.metric_name + "_penalty_{}".format(
            wrong_penalty_rate,
        )
        output_path = join(
            output_dir,
            enriched_metric_name,
        )
        with SimpleTimer("Evaluating [{}]".format(enriched_metric_name)):
            overlapping_scores = []
            for pred, label in zip(predictions, labels):
                overlapping_scores.append(
                    self.evaluate(
                        y_pred=pred,
                        y_true=label,
                        wrong_penalty_rate=wrong_penalty_rate,
                    ),
                )
        with SimpleTimer(
            "Saving [{}] result to {}".format(
                enriched_metric_name,
                output_path,
            ),
        ):

        self.save(
            output_path=output_path,
            annotated_utterances=annotated_utterances,
            overlapping_scores=overlapping_scores,
            wrong_penalty_rate=wrong_penalty_rate,
        )

    def evaluate(
            self,
            y_pred: List[str],
            y_true: List[str],
            wrong_penalty_rate: float = 2.0,
        ) -> float:
        if len(y_pred) != len(y_true):
            raise ValueError(
                "Prediction and label must have same length!!!",
            )
        penalty = 0.0
        for token_p, token_t in zip(y_pred, y_true):
            if token_p == token_t:
                penalty += 0.0
            elif (token_p in ["DONT_CARE", "DONT_CARE*"]) or (
                token_t in ["DONT_CARE", "DONT_CARE*"]):
                penalty += 1.0
            else:
                penalty += wrong_penalty_rate
        overlapping_score = 1 - penalty / len(y_pred)
        return overlapping_score

    def _gen_report(
            self,
            annotated_utterances: List[str],
            overlapping_scores: List[float],
        ) -> pd.DataFrame:
        report_df = pd.DataFrame()
        report_df["annotated_utterances"] = annotated_utterances
        report_df["overlapping_scores"] = overlapping_scores
        return report_df


    def save(
            self,
            output_path: str,
            annotated_utterances: List[str],
            overlapping_scores: List[float],
            wrong_penalty_rate: float,
        ):
        mkdir_p(dirname(output_path))
        report_df = self._gen_report(
            annotated_utterances=annotated_utterances,
            overlapping_scores=overlapping_scores,
        )
        with SimpleTimer(
            "Saving [{}] result to {}".format(
                self.metric_name,
                output_path,
            ),
        ):
            report_df.to_csv(
                path_or_buf=output_path,
                index=False,
            )
