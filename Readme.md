# Machine learning with tensorflow
# Introduction
## Types of machine learning

### Classification
A classification ML-algorithm is one which determines if something belongs to one or more groups.

ML-based classifiers differ from traditional "rule-based" classifiers in that:
  * ML-based classifiers are dynamic while rule-based classifiers are static.
  * Rule-based classifiers require experts while ML-based classifiers do not.
  * ML-based classifiers require a corpus of data while rule-based classifiers do not.

**Feature vectors** are the attributes that the ML algorithm focuses on. Each data point is a list, or vector, of features. In a traditional classifier, the experts determine which features to pay attention to.

#### Neural networks 
A neural network is a deep learning representation classification system. A deep learning system is one which decides it's feature vector in part or in whole independently. A deep learning-based binary classifier (seen in below image) requires a corupus of data, a feature selection & classification algorithm which contains multiple neural layers comprised of neurons. The neural layers which interact with the images directly (the Pixels and Object Parts layer which take in the image and output the image to the ML-based classifier in the example) are called visible layers, while the others are known as hidden layers.

![](./markdownImages/neuralNetwork.png) 

The interaction of the individual neural layers and the neurons themselves can be seen in the following image.
![](./markdownImages/neurons.png) 
The difficulty in designing a neural network is in choosing which features to map and how to architect and implement those designs.

### Regression
Regression, or fitting a curve to a set of data points, is the simplest example of machine learning possible. Linear regression is the simplest form of regression. 

The error of regression is calculated by taking the sum of the squares of the residuals, or distance between the regression fit line and all of the data points. 
### Clustering
### Rule extraction

## Computation graph
Analysis of a computation graph for two distinct nodes, which contain no overlapping dependencies, can be parallelized and even distributed to multiple machines in a cluster or cloud computing application. This can be referred to as "lazy evaluation".

## Tensors
The rank of a tensor is an integer. A scalar, for example, is rank 0.

The two steps in a TensorFlow program are:
  * building a graph
  * running a graph


Tensors connect nodes in a tensorflow application. A tensor is the central unit of data in TensorFlow. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. 

Vectors, which are 1-Dimensional tensor, are defined with one set of square brackets: [1, 3, 5, 7, 9].
A matrix is a 2-Dimensional tensor, which is denoted by two sets of square brackets: [[1, 2], [2, 3] [3, 4 ]].
### Rank of a tensor
The rank, therefore, can be thought of as the number of square brackets enclosing the set of numbers.
### Shape of a tensor
The shape of a tensor defines how many elements exist along a certain dimension. For example, the above matrix ([[1, 2], [2, 3] [3, 4 ]]) has a shape of [2, 3] (three sets of pairs).
### Data type  of a tensor
The final defining characteristic is the data type: int, float, bool, etc.
