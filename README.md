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
from ynlu import NLUClient

client = NLUClient(token='YOUR_TOKEN_HERE')

# Get all possible clf ids
ids = client.get_all_available_clf_ids()
print(ids)

# Get model by id
model = client.get_model_by_id(ids[0])

# Predict
intent_result, entity_result = model.predict('飲料喝到飽')

# Also could get the clf by clf's name
# Get all possible clf names
names = clf.get_all_available_clf_names()
print(names)

# Get model by name
model = client.get_model_by_name(names[0])

# Predict
intent_result, entity_result = model.predict('飲料喝到飽')
```
