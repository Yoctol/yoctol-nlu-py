import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

from .global_variables import INTENT_CLASSIFIER_HOST
from .global_variables import INTENT_CLASSIFIER_URL


class IntentClassifierClient(object):
    '''Client for intent classification.'''

    def __init__(self, token, expected_retries=1):
        self.token = token
        self.classifier_id = None

        # request = requests.get(
        #     INTENT_CLASSIFIER_URL,
        #     headers={
        #         'Host': INTENT_CLASSIFIER_HOST,
        #         'Accept': 'text/html',
        #     },
        # )
        # request.raise_for_status()
        # self.csrf = request.cookies['csrftoken']
        # self._transport = RequestsHTTPTransport(
        #     url=INTENT_CLASSIFIER_URL,
        #     # auth=self.token,
        #     cookies={'csrftoken': self.csrf},
        #     headers={'x-csrftoken': self.csrf},
        #     use_json=True,
        # )

        self._transport = RequestsHTTPTransport(
            url=INTENT_CLASSIFIER_URL,
            use_json=True,
        )
        self._client = Client(
            retries=expected_retries,
            transport=self._transport,
            fetch_schema_from_transport=True,
        )

    def create_classifier(self, name):
        get_all_clf_query = """
            mutation _ {{
              useToken(token: \"{0}\") {{
                ok
              }}
              myClassifiers {{
                edges {{
                  node {{
                    id
                    name
                  }}
                }}
              }}
            }}
        """.format(self.token)
        query = gql(get_all_clf_query)
        result = self._client.execute(
            query,
        )

        all_clf_names = { res['node']['name']: res['node']['id'] for res in result['myClassifiers']['edges']}
        if name not in all_clf_names.keys():
            raw_query = """
                mutation _($input: CreateClassifierInput!) {{
                    useToken(token: \"{0}\") {{
                        ok
                    }}
                    createClassifier(input: $input) {{
                        classifier {{
                            id
                        }}
                    }}
                }}
            """.format(self.token)
            query = gql(raw_query)

            variable_values = {
                'input': {
                    'name': name,
                }
            }

            result = self._client.execute(
                query,
                variable_values=variable_values,
            )
            print(result)

            self.classifier_id = result['createClassifier']['classifier']['id']
        else:
            self.classifier_id = all_clf_names[name]

    def get_classifiers(self):
        pass

    def delete_data(self):
        """Delete all data of the classifier.

        Raises:
            Exception: if classifier_id is None, a exception will raise.

        """
        if self.classifier_id is None:
            raise Exception('Should set classifier id first.')
        get_all_intent_query = """
            mutation _ {{
              useToken(token: \"{0}\") {{
                ok
              }}
              classifier(id: \"{1}\") {{
                intents {{
                  edges {{
                    node {{
                      id
                    }}
                  }}
                }}
              }}
            }}
        """.format(self.token, self.classifier_id)
        query = gql(get_all_intent_query)
        result = self._client.execute(
            query,
        )
        intents = result['classifier']['intents']['edges']
        intent_ids = [intent['node']['id'] for intent in intents]

        for intent_id in intent_ids:
            get_all_intent_query = """
                mutation _($input: DeleteIntentInput!) {{
                    useToken(token: \"{0}\") {{
                        ok
                    }}
                    deleteIntent(input: $input) {{
                        classifier {{
                            id
                        }}
                    }}
                }}
            """.format(self.token)
            query = gql(get_all_intent_query)
            variable_values = {
                'input': {
                    'intentId': intent_id,
                }
            }
            self._client.execute(
                query,
                variable_values=variable_values,
            )

    def set_classifier(self, classifier_id):
        self.classifier_id = classifier_id

    def classifier_is_training(self):
        """Check if the classifier is training.

        """
        if self.classifier_id is None:
            raise ValueError('classifier id is None!')
        raw_query = """
            mutation _ {{
                useToken(token: \"{0}\") {{
                    ok
                }}
                classifier(id: \"{1}\") {{
                    isTraining
                }}
            }}
        """.format(self.token, self.classifier_id)
        query = gql(raw_query)

        result = self._client.execute(
            query,
        )
        return result['classifier']['isTraining']

    def _classifier_valid(self):
        pass

    def create_intents(self, intent_names):
        assert isinstance(intent_names, list)
        for intent in intent_names:
            assert isinstance(intent, str)
        raw_query = """
            mutation _($input: CreateIntentsInput!) {{
                useToken(token: \"{0}\") {{
                    ok
                }}
                createIntents(input: $input) {{
                    intents {{
                        edges {{
                            node {{
                                id
                                name
                            }}
                        }}
                    }}
                }}
            }}
        """.format(self.token)
        query = gql(raw_query)

        variable_values = {
            'input': {
                'classifierId': self.classifier_id,
                'intentNames': intent_names,
            }
        }

        result = self._client.execute(
            query,
            variable_values=variable_values,
        )

    @property
    def intents(self):
        raw_query = """
            mutation _ {{
                useToken(token: \"{0}\") {{
                    ok
                }}
                classifier(id: \"{1}\") {{
                    id
                    intents {{
                      edges {{
                        node {{
                          id
                          name
                          utterances {{
                            edges {{
                              node {{
                                id
                                text
                              }}
                            }}
                          }}
                        }}
                      }}
                    }}
                }}
            }}
        """.format(
            self.token,
            self.classifier_id,
        )
        query = gql(raw_query)

        result = self._client.execute(
            query,
        )

        intents =  [r['node'] for r in result['classifier']['intents']['edges']]
        for intent in intents:
            intent['utterances'] = [u['node'] for u in intent['utterances']['edges']]
        return intents

    def create_utterances(self, intent_id, utterance_texts):
        assert isinstance(intent_id, str)
        assert isinstance(utterance_texts, list)
        for utterance in utterance_texts:
            assert isinstance(utterance, str)
        raw_query = """
            mutation _($input: CreateUtterancesInput!) {{
                useToken(token: \"{0}\") {{
                    ok
                }}
                createUtterances(input: $input) {{
                    utterances {{
                        edges {{
                            node {{
                                id
                                text
                            }}
                        }}
                    }}
                }}
            }}
        """.format(self.token)
        query = gql(raw_query)

        variable_values = {
            'input': {
                'intentId': intent_id,
                'utteranceTexts': utterance_texts,
            }
        }

        result = self._client.execute(
            query,
            variable_values=variable_values,
        )

    def get_utterances(self):
        pass

    def add_intent_utterance_pairs(self, iupairs):
        current_intents = self.intents
        current_intent_names = set(intent['name'] for intent in current_intents)

        intent_names = set(map(lambda x: x['intent'], iupairs))

        new_intent_names = list(intent_names - current_intent_names)
        print('new intents: {}'.format('|'.join(new_intent_names)))
        intent_ids = self.create_intents(new_intent_names)

        current_intents = self.intents
        current_intent_name2id = {
            intent['name']: intent
                for intent in current_intents
        }

        for intent in intent_names:
            intent_id = current_intent_name2id[intent]['id']
            iupair_of_this_intent = [iupair for iupair in iupairs if iupair['intent'] == intent]
            utterances_of_this_intent = set(map(
                lambda x: x['utterance'],
                iupair_of_this_intent,
            ))

            old_utterances_of_this_intent = set(map(
                lambda x: x['text'],
                current_intent_name2id[intent]['utterances']
            ))

            new_utterances = list(utterances_of_this_intent - old_utterances_of_this_intent)
            print('for intent {0}, new utterances: {1}'.format(
                intent,
                ','.join(new_utterances)
            ))
            self.create_utterances(intent_id, new_utterances)

    def get_intent_utterance_pairs(self):
        pass

    def train(self):
        raw_query = """
            mutation _($input: TrainClassifierInput!) {{
                useToken(token: \"{0}\") {{
                    ok
                }}
                trainClassifier(input: $input) {{
                    classifier {{
                        id
                    }}
                }}
            }}
        """.format(self.token)
        query = gql(raw_query)

        variable_values = {
            'input': {
                'classifierId': self.classifier_id,
            }
        }

        result = self._client.execute(
            query,
            variable_values=variable_values,
        )

    def predict(self, text):
        assert isinstance(text, str)
        raw_query = """
            mutation _($input: PredictInput!) {{
                useToken(token: \"{0}\") {{
                    ok
                }}
                predict(input: $input) {{
                    classifier {{
                        id
                    }}
                    predictions {{
                      edges {{
                        node {{
                          intent {{
                            id
                            name
                          }}
                          score
                        }}
                      }}
                    }}
                }}
            }}
        """.format(self.token)
        query = gql(raw_query)

        variable_values = {
            'input': {
                'classifierId': self.classifier_id,
                'text': text
            }
        }

        result = self._client.execute(
            query,
            variable_values=variable_values,
        )

        parsed_result = result['predict']['predictions']['edges']
        parsed_result = [res['node'] for res in parsed_result]
        parsed_result = [
            {
                'intent': ans['intent']['name'],
                'score': ans['score'],
            }
            for ans in parsed_result
        ]

        return sorted(parsed_result, key=lambda x: x['score'], reverse=True)
