from typing import List, Dict
import re

from ynlu.sdk.evaluation import NOT_ENTITY, UNKNOWN


SUB_PROG = re.compile(r"<[^\<\>]*?>(?P<KEEP>[^\<\>]*?)</[^\<\>]*?>")
FINDALL_PROG = re.compile(r"<([^\<\>]*?)>([^\<\>]*?)</")


def remove_annotation(annotated_utterance: str):
    return SUB_PROG.sub("\g<KEEP>", annotated_utterance)


def preprocess_annotated_utterance(
        annotated_utterance: str,
        not_entity: str = NOT_ENTITY,
    ) -> List[str]:
    """Character Level Entity Label Producer

        Named-entity of each character is extracted by XML-like annotation.
        Also, they would be collected in a list conform to the order of characters
        in the sentence.

        Args:
            annotated_utterance (a string):
                An utterance with annotations looks like <a>blabla</a>.
                It is a special format for labeling named-entity in an utterance.
            not_entity (a string, default = "DONT_CARE"):
                A representation of words that we don't care about.

        Returns:
            entities (a list of string):
                A list of named-entity labels in character level.

        Examples:
            >>> from ynlu.sdk.evaluation.utils import preprocess_annotated_utterance
            >>> preprocess_annotated_utterance(
                annotated_utterance="<drink>Coffee</drink>, please.",
                not_entity="n",
            )
            >>> ["drink", "drink", "drink", "drink", "drink", "drink", "n",
                "n", "n", "n", "n", "n", "n", "n", "n"]
    """
    clean_utterance = remove_annotation(annotated_utterance)
    entity_word_pair = FINDALL_PROG.findall(annotated_utterance)
    entities = [not_entity] * len(clean_utterance)
    begin_index = 0
    for entity, word in entity_word_pair:
        start_idx = clean_utterance.find(word, begin_index)
        if start_idx == -1:
            raise ValueError(
                "Word {} can not be found in {}".format(word, clean_utterance),
            )
        entities[start_idx: start_idx + len(word)] = [entity] * len(word)
        begin_index = start_idx + len(word)
    return entities


def preprocess_entity_prediction(
        utterance: str,
        entity_prediction: List[dict],
        not_entity: str = NOT_ENTITY,
    )-> List[str]:
    """Character Level Entity Label Producer

        Named-entity of each character is extracted by the output of model prediction.
        Also, they would be collected in a list conform to the order of characters
        in the sentence.

        Args:
            utterance (a string):
                An input of `model.predict()`.
            entity_prediction (a list dictionaries):
                Entity part of output returns by calling `model.predict()` with
                the utterance above as input. The element in the list is a segment
                of utterance, the predicted entity type and the confidence of
                prediction.
            not_entity (a string, default = "DONT_CARE"):
                A representation of words that we don't care about.

        Returns:
            entities (a list of string):
                A list of named-entity labels in character level.

        Examples:
            >>> from ynlu.sdk.evaluation.utils import preprocess_annotated_utterance
            >>> preprocess_entity_prediction(
                    utterance="Coffee, please.",
                    entity_prediction=[
                        {"value": "Coffee", "entity": "drink", "score": 0.8},
                        {"value": ", please.", "entity": "n"}, "score": 0.7},
                    ],
                    not_entity="n",
                )
            >>> ["drink", "drink", "drink", "drink", "drink", "drink", "n",
                "n", "n", "n", "n", "n", "n", "n", "n"]
    """
    entities = [not_entity] * len(utterance)
    begin_index = 0
    for pred in entity_prediction:
        if begin_index >= len(utterance):
            break
        for char in pred["value"]:
            start_idx = utterance.find(char, begin_index)
            if start_idx == -1:
                raise ValueError(
                    "Word {} can not be found in {}".format(char, utterance),
                )
            entities[start_idx] = pred["entity"]
            begin_index = start_idx + len(char)
    return entities


def preprocess_intent_prediction_by_threshold(
        intent_predictions: List[List[Dict[str, str]]],
        threshold: float = 0.5,
        unknown_token: str = UNKNOWN,
    ) -> List[str]:
    """Predicted Intent Overrider

        Override the predicted intent with the unknown token when its score is
        lower than the threshold.

        Args:
            intent_predictions( a list of intent predictions):
                A list of intent part of output by calling `model.predict()`.
            threshold (float, defualt is 0.5):
                The indicator about whether to override the prediciton or not.
            unknown_token (a string, default is UNk):
                A token which would be used as an alternative to a lower-confidence
                intent prediction.

        Returns:
            output (a list of string):
                A list of preprocessed intent labels.

        Examples:
            >>> from ynlu.sdk.evaluation.utils import preprocess_intent_prediction_by_threshold
            >>> preprocess_intent_prediction_by_threshold(
                    intent_predictions=[
                        [{"intent": "a", "score": 0.3}, {"intent": "b", "score": 0.1}],
                        [{"intent": "b", "score": 0.7}, {"intent": "c", "score": 0.3}],
                    ]
                    threshold=0.5,
                    unknown_token="oo",
                )
            >>> [["oo", "oo"], ["b", "oo"]]
    """
    output = []
    for intent_pred in intent_predictions:
        if intent_pred[0]["score"] > threshold:
            output.append(intent_pred[0]["intent"])
        else:
            output.append(unknown_token)
    return output
