from typing import List

from gql import Client
from gql.transport.requests import RequestsHTTPTransport

from .model import Model


class NLUClient(object):
    """
    Client which could contain multiple intent clfs
    NOTE: Only support predicting for now
    """

    URL = 'https://ynlu.yoctol.com/graphql'

    def __init__(
        self,
        token: str,
        classifier_ids: List[str],
        expected_retries: int = 1,
        url: str = URL,
    ):
        self.token = token

        self._transport = RequestsHTTPTransport(
            url=url,
            use_json=True,
        )
        self._client = Client(
            retries=expected_retries,
            transport=self._transport,
            fetch_schema_from_transport=True,
        )
        # Remove duplication
        self._classifier_ids = list(set(classifier_ids))
        self._models = {
            clf_id: Model(clf_id, self._client) for clf_id in classifier_ids
        }

    def __getitem__(self, key):
        return self.get_model_by_id(key)

    def get_model_by_id(
        self,
        classifier_id: str,
    ) -> Model:
        self.check_clf_id(classifier_id)
        return self._models[classifier_id]

    def check_clf_id(
        self,
        classifier_id: str,
    ) -> None:
        if classifier_id not in self._classifier_ids:
            raise ValueError('Illegal clf id {}'.format(classifier_id))
