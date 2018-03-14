# yoctol-nlu-py
Yoctol Natural Language Understanding SDK for python.

## Install
Use Python3
```
pip install yoctol-nlu
```

## Usage

### Intent Classifier Service

For existing classifier:
```python
from time import sleep
from ynlu import NLUClient

client = NLUClient(
    token='TOKEN_HERE',
    classifier_ids=['clf_id1', 'clf_id2', 'more_clf_ids']
)

# get the model given the classifier_id
model = client['clf_id1']

intent_result, entity_result = model.predict('飲料喝到飽')

```
