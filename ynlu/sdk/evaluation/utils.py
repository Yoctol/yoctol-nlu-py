from typing import List
import re

from ynlu.sdk.evaluation import NOT_ENTITY


SUB_PROG = re.compile(r"<[^\<\>]*?>(?P<KEEP>[^\<\>]*?)</[^\<\>]*?>")
FINDALL_PROG = re.compile(r"<([^\<\>]*?)>([^\<\>]*?)</")


def remove_annotation(annotated_utterance: str):
    return SUB_PROG.sub("\g<KEEP>", annotated_utterance)


def preprocess_annotated_utterance(
        annotated_utterance: str,
        not_entity: str = NOT_ENTITY,
    ) -> List[str]:
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
