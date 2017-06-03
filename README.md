# yoctol-nlu-py
Yoctol Natural Language Understanding SDK for python.

## Install
Use Python3.
```
pip install yoctol-nlu
```

## Usage

### Intent Classifier Service

For new user:
```python
from time import sleep
from ynlu import IntentClassifierClient

client = IntentClassifierClient(
    token='TOKEN',
)

# create a classifier
# If the name exist, will use the existed one.
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

client.train() # async train

while True:
    if client.classifier_is_training():
        sleep(1)
        continue
    break

result = client.predict('你好嗎') # This is a action without side-effects
'''
>>> result
[{'score': 0.7828801870346069, 'intent': '打招呼'}, 
 {'score': 0.11771836876869202, 'intent': '感謝'}, 
 {'score': 0.0994015485048294, 'intent': '說再見'}]
'''
```

For existing classifier:
```python
from ynlu import IntentClassifierClient

client = IntentClassifierClient(
    token='TOKEN',
)

client.set_classifier(classifier_id='CLASSIFIER_ID')

result = client.predict('你好嗎')
```
