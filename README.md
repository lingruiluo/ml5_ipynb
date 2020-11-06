# ml5_ipynb

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lingruiluo/ml5_ipynb.git/HEAD)

Wraps up ml5.js for Jupyter interface

## Goal

`ml5_ipynb` is designed to allow implementation of the Javascript module [`ml5.js`](https://ml5js.org/) on Jupyter interface for a faster training of simple neural network models and pre-train models without GPU/TPU.

## Install

### For development
```
pip install -e .
```

### For user
```
pip install git+https://github.com/lingruiluo/ml5_ipynb.git
```

## Introduction

[`ml5.js`](https://ml5js.org/) is a web-based machine learning and deep learning tool that aims to provide access to machine learning and deep learning models. Also, it is a web establishment for presenting machine learning and deep learning training processes and results. It is built on top of tensorflow.js and it can handle GPU-accelerated operations for models.   
The module `ml5_ipynb` is a Jupyter widget version of ml5.js that provides a possibility of implementing machine learning and deep learning models supported by ml5.js on Jupyter interface. The advantage of this module is to allow a bit faster training and predicting of simple models using local machine without GPU/TPU on Jupyter notebook.

### Current support models

#### Neural network

The neural network class `neuralNetwork` supports three types of deep learning tasks:
  - Regression  
  - Classification   
  - Image Classification (TODO)     
The type of task can be specified in the `options` before initializing the network. A simple way to declare the network is shown as followed.
```python
nn = ml5_ipynb.ml5_nn.neuralNetwork()
```

#### Image Classification
