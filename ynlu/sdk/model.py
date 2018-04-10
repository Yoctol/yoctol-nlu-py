from typing import List, Dict, Tuple
import re

from gql import Client, gql


class Model(object):

    def __init__(
            self,
            classifier_id: str,
            client: Client,
        ):
        if not bool(re.match(r'^[0-9]+$', classifier_id)):
            raise ValueError(
                'id should be string of number, got {}'.format(
                    classifier_id,
                ),
            )
        self._classifier_id = classifier_id
        self._client = client

    @property
    def model_id(self):
        return self._classifier_id

    def train(self):
        raise NotImplementedError('Not for now')

    def predict(
            self,
            utterance: str,
            exactly: bool = True,
        ) -> Tuple[List[Dict], List[Dict]]:
        if not isinstance(utterance, str):
            raise ValueError(
                'utterance is not str, got {}'.format(
                    type(utterance),
                ),
            )
        raw_query = """
            mutation predict($classifierId: String!, $text: String!, $exactly: Boolean) {
                predict(classifierId: $classifierId, text: $text, exactly: $exactly) {
                    intents {
                        name
                        score
                    }
                    entities {
                        name
                        value
                        score
                    }
                    match {
                        isMatched
                        score
                    }
                }
            }
        """
        gql_query = gql(raw_query)
        variable_values = {
            'classifierId': self._classifier_id,
            'text': utterance,
            'exactly': exactly,
        }
        result = self._client.execute(
            gql_query,
            variable_values=variable_values,
        )

        intents_prediction = result['predict']['intents']
        intents_prediction = [
            {
                'intent': ans['name'],
                'score': ans['score'],
            }
            for ans in intents_prediction
        ]

        entities_prediction = result['predict']['entities']
        entities_prediction = [
            {
                'entity': ans['name'],
                'value': ans['value'],
                'score': ans['score'],
            }
            for ans in entities_prediction
        ]

        return intents_prediction, entities_prediction

    def batch_predict(
            self,
            utterances: List[str],
        ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        intents_predictions, entities_predictions = zip(*[self.predict(utt) for utt in utterances])
        return list(intents_predictions), list(entities_predictions)
