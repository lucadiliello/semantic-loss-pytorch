# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='semantic_loss_pytorch',
    version='0.1', # get from pypsdd.__init__
    description='Semantic loss function for PyTorch based on PySDD',
    long_description=readme,
    author='Jacopo Gobbi & Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    url='http://reasoning.cs.ucla.edu/psdd',
    python_requires='>=3.5',
    license=license,
    packages=["py3psdd"]
)
