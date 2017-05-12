# yoctol-nlu-py
Yoctol Natural Language Understanding SDK for python.

## Install
```
pip install yoctol-nlu
```

## Usage

### Intent Classifier Service

For new user:
```python
from ynlu import IntentClassifierClient

client = IntentClassifierClient(
    token='TOKEN',
)

client.create_classifier(
    name='clf_for_test'
)

# to get the classifier id:
# print(client.classifier_id)

# create intent, utterances pairs
# This is a idempotent action, since it will check every item if it is added before.
client.add_intent_utterance_pairs([
    {'intent': '打招呼', 'utterance': '嗨'},
    {'intent': '感謝', 'utterance': '謝謝'},
    {'intent': '說再見', 'utterance': '再見'},
    {'intent': '打招呼', 'utterance': '早安'},
    {'intent': '打招呼', 'utterance': '你好'},
    {'intent': '感謝', 'utterance': '非常感謝'},
    {'intent': '說再見', 'utterance': '掰掰'},
    {'intent': '感謝', 'utterance': '有你真好'},
    {'intent': '說再見', 'utterance': '下次見'},
]) 

client.train()

# wait a minute...
result = client.predict('你好嗎') # This is a action without side-effects
```

For existing classifier:
```python
from ynlu import IntentClassifierClient

client = IntentClassifierClient(
    token='TOKEN',
)

client.get_classifier(classifier_id='CLASSIFIER_ID')

result = client.predict('你好嗎')
```
