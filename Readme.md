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
### Clustering
### Rule extraction

