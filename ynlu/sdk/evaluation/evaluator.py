from typing import List, Dict
from mkdir_p import mkdir_p
import logging

from .utils import (
    remove_annotation,
    preprocess_annotated_utterance,
    preprocess_entity_prediction,
)
from .intent_evaluators import INTENT_EVALUATORS
from .entity_evaluators import ENTITY_EVALUATORS

LOGGER = logging.getLogger(__name__)


class Evaluator(object):

    def __init__(
            self,
            utterances: List[str],
            result_output_dir: str,
            intents: List[str] = None,
            annotated_utterances: List[str] = None,
        ):
        if intents is not None:
            if len(utterances) != len(intents):
                raise ValueError(
                    """
                        Number of utterances [{}] is not equal to
                        that of intents [{}].
                    """.format(
                        len(utterances),
                        len(intents),
                    ),
                )
        if annotated_utterances is not None:
            if len(annotated_utterances) != len(utterances):
                raise ValueError(
                    """
                        Number of annotated utterances [{}] is not equal to
                        that of utterances [{}].
                    """.format(
                        len(annotated_utterances),
                        len(utterances),
                    ),
                )

            entity_labels = []
            for utt, anno_utt in zip(utterances, annotated_utterances):
                if utt != remove_annotation(anno_utt):
                    raise ValueError(
                        """
                           Annotated utterance {} is not the same as utterance {},
                           after removing annotations.
                        """.format(
                            anno_utt,
                            utt,
                        ),
                    )
                else:
                    entity_labels.append(
                        preprocess_annotated_utterance(
                            utterance=utt,
                            annotated_utterance=anno_utt,
                        ),
                    )

        self.utterances = utterances
        self.annotated_utterances = annotated_utterances
        self.intents = intents
        self.entity_labels = entity_labels
        self.result_output_dir = result_output_dir
        mkdir_p(self.result_output_dir)

    @classmethod
    def get_all_intent_evaluation_methods(cls,):
        return INTENT_EVALUATORS.keys()

    @classmethod
    def get_all_entity_evaluation_methods(self):
        return ENTITY_EVALUATORS.keys()

    def evaluate_intent(
            self,
            intent_predictions: List[Dict[str, str]],
            evaluation_methods: List[List[str, dict]],
        ):
        if self.intents is None:
            LOGGER.warning(
                "intent labels are not provided. @.@",
            )
            return None

        if len(self.intents) != len(intent_predictions):
            raise ValueError(
                """
                    Number of intents [{}] is not equal to
                    that of intent predictions [{}].
                """.format(
                    len(self.intents),
                    len(intent_predictions),
                ),
            )
        for eval_method, eval_params in evaluation_methods:
            eval_func = globals()[eval_method]
            if eval_func is not None:
                eval_func(
                    y_pred=intent_predictions,
                    y_true=self.intents,
                    output_dir=self.result_output_dir,
                    **eval_params,
                )
            else:
                LOGGER.warning(
                    "[{}] is not an intent evaluation method".format(
                        eval_method),
                )

    def evaluate_entity(
            self,
            utterances: List[str],
            entity_predictions: List[Dict[str, str]],
            evaluation_methods: List[str, Dict[str, str]],
        ):
        if len(self.annotated_utterances) != len(entity_predictions):
            raise ValueError(
                """
                    Number of annotated utterances [{}] is not equal to
                    that of entity predictions [{}].
                """.format(
                    len(self.annotated_utterances),
                    len(entity_predictions),
                ),
            )
        predictions = []
        for utterance, pred in zip(utterances, entity_predictions):
            predictions.append(
                preprocess_entity_prediction(
                    utterance=utterance,
                    entity_prediction=pred,
                ),
            )

        for eval_method, eval_params in evaluation_methods:
            eval_func = globals()[eval_method]
            if eval_func is not None:
                eval_func(
                    y_pred=entity_predictions,
                    y_true=self.entity_labels,
                    output_dir=self.result_output_dir,
                    **eval_params,
                )
            else:
                LOGGER.warning(
                    "[{}] is not an entity evaluation method".format(
                        eval_method,
                    ),
                )
