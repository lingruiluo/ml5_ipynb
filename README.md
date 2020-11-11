# ml5_ipynb

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lingruiluo/ml5_ipynb.git/HEAD)

Wraps up ml5.js for Jupyter interface

## Goal

`ml5_ipynb` is designed to allow implementation of the Javascript module [`ml5.js`](https://ml5js.org/) on Jupyter interface for a faster training of simple neural network models and pre-train models with remote GPU.

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

Let's briefly introduce the current models provided by ml5_ipynb. More details will be introducted in the [tutorial](https://github.com/lingruiluo/ml5_ipynb/blob/main/ml5_ipynb%20Tutorial.ipynb).

#### Neural network

The neural network class `neuralNetwork` supports three types of deep learning tasks:
  - Regression  
  - Classification   
  - Image Classification (TODO)     
The type of task can be specified in the `options` before initializing the network. A simple way to declare the network is shown as followed.
```python
nn = ml5_ipynb.ml5_nn.neuralNetwork()
```
Examples include [`Neural Network Simple Examples`](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/Neural%20Network%20Simple%20Examples.ipynb), [`color classification`](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/Color%20Classification.ipynb), [`CO2 Emission Example with multi-layers NN`](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/CO2%20Emission%20Example%20with%20multi-layers%20NN.ipynb).

#### Image Classification

The image classification method is designed to classify an image using pre-trained models including `MobileNet`, `Darknet`, `DoodleNet` and any other saved models. 
  - Use MobileNet model
    ```python
    nn = ml5_ipynb.ml5_image.imageClassifier('MobileNet')
    ```   
  - Use saved model
    ```python
    path = 'user/mymodel.json'
    nn = ml5_ipynb.ml5_image.imageClassifier(path)
    ```
Examples include [`Image Classifier`](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/Image%20Classifier.ipynb).

#### KMeans

ml5_ipynb also has some methods for machine learning models. Currently, we only have kmeans model but will support more later on. Kmeans is a cluster method that can be used for many tasks including image segmentation and object detection. This method is not a best choice for some of tasks such as object detection since it gives different results each time we initialize, however, it can still work.   
In the [example](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/kmeans%20simple%20example.ipynb) we have here, kmeans uses default number of clusters which is 3 to cluster the image.  
<p align="center">
  <img src="examples/pic/faces.jpg" width="250" height="150" title="Original image">
  <img src="examples/pic/clustered.jpg" width="250" height="150" title="Clustered image">
</p>

A simple way to create kmean class is 
        ```
        nn = ml5_kmeans.Kmeans()
        ```
There is an example [`kmeans simple example`](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/kmeans%20simple%20example.ipynb).

#### Object Detection

Object detection method uses YOLO or CocoSsd model. 

```python
# using YOLO
nn = ml5_detector.ObjectDetector('yolo')
```
There is an example [`Object Detection using YOLO`](https://github.com/lingruiluo/ml5_ipynb/blob/main/examples/Object%20Detection%20using%20YOLO.ipynb).
