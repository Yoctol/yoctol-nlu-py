from unittest import TestCase

from uttut.elements import Intent, Entity

from ..model import Model


FAKE_QUERY = {
    'predict': {
        'intents': [
            {
                'name': '測試2',
                'score': 0.78,
            },
            {
                'name': '測試1',
                'score': 0.17,
            },
        ],
        'entities': [
            {
                'name': 'PLACE',
                'value': '台北',
                'score': 0.8877,
            },
            {
                'name': 'TEMP',
                'value': '37度C',
                'score': 0.7788,
            },
        ],
    },
}
FAKE_INTENT_SHOULD_RETURN = [
    {
        'intent': Intent('測試2'),
        'score': 0.78,
    },
    {
        'intent': Intent('測試1'),
        'score': 0.17,
    },
]
FAKE_ENTITY_SHOULD_RETURN = [
    {
        'entity': Entity(
            name='PLACE',
            value='台北',
            start=0,
            end=0,
        ),
        'score': 0.8877,
    },
    {
        'entity': Entity(
            name='TEMP',
            value='37度C',
            start=0,
            end=0,
        ),
        'score': 0.7788,
    },
]


class MockClient:

    def execute(self, gql_query, variable_values):
        return FAKE_QUERY


class ModelTestCase(TestCase):

    def setUp(self):
        self.model = Model(
            classifier_id='7878',
            client=MockClient(),
        )

    def test_model_init(self):
        self.assertEqual(self.model.model_id, '7878')

    def test_model_id(self):
        self.assertEqual(self.model.model_id, self.model._classifier_id)

    def test_model_predict(self):
        with self.assertRaises(ValueError):
            illegal_utterance = 11223
            self.model.predict(illegal_utterance)
        str_utterance = 'something_i_dont_know'
        intents_result, entities_result = self.model.predict(str_utterance)
        self.assertEqual(FAKE_INTENT_SHOULD_RETURN, intents_result)
        self.assertEqual(FAKE_ENTITY_SHOULD_RETURN, entities_result)
