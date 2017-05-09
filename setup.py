# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from yoctol_nlu import __version__

try:
    long_description = open("README.md").read()
except IOError:
    long_description = ""

setup(
    name="yoctol-nlu",
    version=__version__,
    description="Yoctol Natural Language Understanding SDK",
    license="MIT",
    author="cph",
    packages=find_packages(),
    install_requires=[],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
    ]
)
