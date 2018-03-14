from unittest import TestCase

from ..client import NLUClient


def mock_client(_, __):
    return 'Some random client'


class NLUClientTestCase(TestCase):

    def setUp(self):
        self.token = 'some_token_here'
        self.model_ids = ['5566', 'some_id', 'here', '1212test7878', 'here']
        NLUClient.build_client = mock_client
        self.client = NLUClient(
            token=self.token,
            classifier_ids=self.model_ids,
            url='',
        )

    def test_client_init(self):
        client = self.client
        self.assertEqual(client.token, self.token)
        self.assertEqual(client._classifier_ids, list(set(self.model_ids)))
        for model_id in client._classifier_ids:
            self.assertEqual(model_id, client[model_id].model_id)

    def test_get_model_by_id(self):
        right_id = '5566'
        model = self.client[right_id]
        self.assertEqual(model.model_id, right_id)

        wrong_id = '7788'
        with self.assertRaises(ValueError):
            model = self.client[wrong_id]
