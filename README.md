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

#### Fetch Model & One Line Prediction
```python
from ynlu import NLUClient

client = NLUClient(token='YOUR_TOKEN_HERE')

# Get all possible clf ids
ids = client.get_all_available_clf_ids()
print(ids)

# Get model by id
model = client.get_model_by_id(ids[0])

# Predict
intent_prediction, entity_prediction = model.predict('飲料喝到飽')

# Also could get the clf by clf's name
# Get all possible clf names
names = client.get_all_available_clf_names()
print(names)

# Get model by name
model = client.get_model_by_name(names[0])

# Predict
intent_prediction, entity_prediction = model.predict('飲料喝到飽')
```

#### Evaluations for Intent
```python
from ynlu import NLUClient
from ynlu.sdk.evaluation import {
    # Accuracy
    intent_accuracy_score_with_threshold,
    # Precision
    intent_precision_score_with_threshold,
    # Recall
    intent_recall_score_with_threshold,
}

client = NLUClient(token='YOUR_TOKEN_HERE')
model = client.get_model_by_id('TARGET_MODEL_ID_HERE')

test_data = [
    'This is a line ',
    'for testing the NLUClient ',
    'and evaluating the prediction ',
    'from the trained model. ',
]

intent_predictions, entities_predictions = model.batch_predict(test_data)

# Pure Accuracy
print(intent_accuracy_score_with_threshold(
        intent_predictions=intent_predictions,
        y_trues=test_data,
        threshold=0.,
    )
)
# Accuracy with threshold 0.5
print(intent_accuracy_score_with_threshold(
        intent_predictions=intent_predictions,
        y_trues=test_data,
        threshold=0.5,
    )
)
```

Check out the tutorials for more examples.

## Documentation

We rely on Sphinx for user and API documentation.

You can run just make to do rebuild the API stubs and then build the HTML documentation.

```
cd docs
make # equivalent to `make apidoc && make html`
```

To only build the html pages:

```
cd docs
make html
```

To just re-generate the API reference.

```
cd docs
make apidoc # calls sphinx-apidoc
```
Run `make help` for a full list of build options.
