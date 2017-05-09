class IntentClassifierClient(object):
    '''Client for intent classification.'''

    def __init__(self, token):
        self.token = token
        self.classifier_id = None

    def create_classifier(self):
        pass

    def get_classifiers(self):
        pass

    def set_classifier(self):
        pass

    def _classifier_valid(self):
        pass

    def create_intents(self):
        pass

    def get_intents(self):
        pass

    def create_utterances(self):
        pass

    def get_utterances(self):
        pass

    def add_intent_utterance_pairs(self):
        pass

    def get_intent_utterance_pairs(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
