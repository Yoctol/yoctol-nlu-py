from unittest import TestCase

from ..client import NLUClient


TEST_MODEL_ID = ['5566', '7788', '1024']


def mock_client(_, retries):
    return 'Some random client'

def mock_fetch_all_available_clf_ids(_):
    return TEST_MODEL_ID


class NLUClientTestCase(TestCase):

    def setUp(self):
        self.token = 'some_token_here'
        NLUClient.build_client = mock_client
        NLUClient.fetch_all_available_clf_ids = mock_fetch_all_available_clf_ids
        self.client = NLUClient(
            token=self.token,
            url='',
        )

    def test_client_init(self):
        client = self.client
        self.assertEqual(client.token, self.token)
        self.assertEqual(client._classifier_ids, TEST_MODEL_ID)
        for model_id in client._classifier_ids:
            self.assertEqual(model_id, client[model_id].model_id)

    def test_get_model_by_id(self):
        right_id = TEST_MODEL_ID[0]
        model = self.client[right_id]
        self.assertEqual(model.model_id, right_id)

        wrong_id = '1234567890'
        with self.assertRaises(ValueError):
            model = self.client[wrong_id]
