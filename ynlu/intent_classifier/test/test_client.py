from .. import IntentClassifierClient

FAKE_TOKEN = '1l2kj34iuwy4t8er7yfi8312eiqwueyrt93.9347rywe8o7rfc70y4hxc9gqx89w7ge'

def test_client_init_token():
    client = IntentClassifierClient(token=FAKE_TOKEN)
    assert client.token == FAKE_TOKEN
