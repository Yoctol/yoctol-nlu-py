from typing import List

from gql import gql, Client
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
            expected_retries: int = 1,
            url: str = URL,
        ):
        self.token = token

        self._transport = RequestsHTTPTransport(
            url=url,
            use_json=True,
        )
        self._transport.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; " +
            "Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0",
            "Authorization": "Bearer {}".format(self.token),
            "content-type": "application/json",
        }
        self._client = self.build_client(retries=expected_retries)
        self._classifier_ids, self._classifier_names = self.fetch_all_available_clf_ids_and_names()
        self._models = {
            clf_id: Model(clf_id, self._client) for clf_id in self._classifier_ids
        }

    def fetch_all_available_clf_ids_and_names(self) -> List[str]:
        print("Fetching all classifier's id")
        projects_raw_query = """
            query projects {
                projects {
                    id
                }
            }
        """
        projects_query = gql(projects_raw_query)
        projects_result = self._client.execute(projects_query)
        if 'projects' not in projects_result:
            return []
        projects_id = [projects['id'] for projects in projects_result['projects']]
        clfs_id = []
        clfs_name = []
        for p_id in projects_id:
            clfs_raw_query = """
                query project($id: Int!) {
                    project(id: $id) {
                        classifiers {
                            id
                            name
                        }
                    }
                }
            """
            clfs_query = gql(clfs_raw_query)
            variable_values = {
                'id': p_id,
            }
            clfs_result = self._client.execute(
                clfs_query,
                variable_values=variable_values,
            )
            for clf in clfs_result['project']['classifiers']:
                clfs_id.append(clf['id'])
                clfs_name.append(clf['name'])

        return clfs_id, clfs_name

    def get_all_available_clf_ids(self) -> List[str]:
        return self._classifier_ids[:]

    def get_all_available_clf_names(self) -> List[str]:
        return self._classifier_names[:]

    def build_client(self, retries: int):
        return Client(
            retries=retries,
            transport=self._transport,
            fetch_schema_from_transport=True,
        )

    def __getitem__(self, key):
        return self.get_model_by_id(key)

    def get_model_by_id(
            self,
            classifier_id: str,
        ) -> Model:
        self.check_clf_id(classifier_id)
        return self._models[classifier_id]

    def get_model_by_name(
            self,
            classifier_name: str,
        ) -> Model:
        self.check_clf_name(classifier_name)
        id_index = self._classifier_names.index(classifier_name)
        classifier_id = self._classifier_ids[id_index]
        return self._models[classifier_id]

    def check_clf_id(
            self,
            classifier_id: str,
        ) -> None:
        if classifier_id not in self._classifier_ids:
            raise ValueError('Illegal clf id {}'.format(classifier_id))

    def check_clf_name(
            self,
            classifier_name: str,
        ) -> None:
        if classifier_name not in self._classifier_names:
            raise ValueError('Illegal clf name {}'.format(classifier_name))
