# yoctol-nlu-py
[![Build Status](https://travis-ci.org/Yoctol/yoctol-nlu-py.svg?branch=master)](https://travis-ci.org/Yoctol/yoctol-nlu-py)
[![PyPI version](https://badge.fury.io/py/yoctol-nlu.svg)](https://badge.fury.io/py/yoctol-nlu)

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
