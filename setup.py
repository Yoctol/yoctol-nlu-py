# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

try:
    long_description = open("README.md").read()
except IOError:
    long_description = ""

about = {}
with open(os.path.join(here, "ynlu", "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name="yoctol-nlu",
    version=about["__version__"],
    description="Yoctol Natural Language Understanding SDK",
    license="MIT",
    author="cph",
    packages=find_packages(),
    install_requires=[
        'gql==0.1.0',
        'requests==2.13.0',
    ],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ]
)
