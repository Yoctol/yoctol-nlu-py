from typing import List


class Model(object):

    def __init__(
        self,
        classifier_id: str,
    ):
        self.classifier_id = classifier_id

    def train(self):
        raise NotImplementedError('Not for now')

    def predict(self, utterance: str):
        if not isinstance(utterance, str):
            raise ValueError('utterance is not str, got {}'.format(
                type(utterance)
                )
            )
        # TODO
        pass

    def batch_predict(self, utterances: List[str]):
        predictions = [self.predict(utt) for utt in utterances]
        return predictions
