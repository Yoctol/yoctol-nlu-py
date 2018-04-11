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

Check out the tutorials.

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
