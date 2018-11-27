# Machine learning with tensorflow

# Introduction

## Types of machine learning

### Classification

A classification ML-algorithm is one which determines if something belongs to one or more groups.

ML-based classifiers differ from traditional "rule-based" classifiers in that:

- ML-based classifiers are dynamic while rule-based classifiers are static.
- Rule-based classifiers require experts while ML-based classifiers do not.
- ML-based classifiers require a corpus of data while rule-based classifiers do not.

**Feature vectors** are the attributes that the ML algorithm focuses on. Each data point is a list, or vector, of features. In a traditional classifier, the experts determine which features to pay attention to.

<a name="neural-networks"></a>

### Neural networks

A neural network is a deep learning representation classification system. A deep learning system is one which decides it's feature vector in part or in whole independently. A deep learning-based binary classifier (seen in below image) requires a corupus of data, a feature selection & classification algorithm which contains multiple neural layers comprised of neurons. The neural layers which interact with the images directly (the Pixels and Object Parts layer which take in the image and output the image to the ML-based classifier in the example) are called visible layers, while the others are known as hidden layers.

![](./markdownImages/neuralNetwork.png)

The interaction of the individual neural layers and the neurons themselves can be seen in the following image.
![](./markdownImages/neurons.png)
The difficulty in designing a neural network is in choosing which features to map and how to architect and implement those designs.

### Regression

Regression, or fitting a curve to a set of data points, is the simplest example of machine learning possible. Linear regression is the simplest form of regression.

The error of regression is calculated by taking the sum of the squares of the residuals, or distance between the regression fit line and all of the data points.

In the LinearRegressionWithFakeData.ipynb example, the GradientDescentOptimizer library is used to solve the linear regression on a very simple data set to find the best fit line. Optimizer libraries provide simple, out of the box solutions to regression.

### Clustering

### Rule extraction

## Computation graph

A computation graph, is a directed, acyclic representation of a TensorFlow algorithm. The tensors, the arrows in the graph, flow strictly from left to right and are modified by the nodes within the graph.

![](./markdownImages/computationGraph.png)

The connections to the left of a node are known as the direct or indirect dependencies.

![](./markdownImages/computationGraphDependencies.png)

The result of the computation graph has the loss calculated, which is then fed back in to the neural network which can be visualized using the neural network layered diagram:

![](.\markdownImages\neuralNetworkFeedback.png)

or the computation graph via "unrolling" the graph:

![](.\markdownImages\computationGraphFeedback.png)

Analysis of a computation graph for two distinct nodes, which contain no overlapping dependencies, can be parallelized and even distributed to multiple machines in a cluster or cloud computing application. This can be referred to as "lazy evaluation".

## Tensors

The rank of a tensor is an integer. A scalar, for example, is rank 0.

The two steps in a TensorFlow program are:

- building a graph
- running a graph

Tensors connect nodes in a computation graph. A tensor is the central unit of data in TensorFlow. A tensor consists of a set of primitive values shaped into an array of any number of dimensions.

Vectors, which are 1-Dimensional tensor, are defined with one set of square brackets: [1, 3, 5, 7, 9].
A matrix is a 2-Dimensional tensor, which is denoted by two sets of square brackets: [[1, 2], [2, 3][3, 4 ]].

### Rank of a tensor

The rank, therefore, can be thought of as the number of square brackets enclosing the set of numbers.

### Shape of a tensor

The shape of a tensor defines how many elements exist along a certain dimension. For example, the above matrix ([[1, 2], [2, 3][3, 4 ]]) has a shape of [2, 3] (three sets of pairs).

### Data type of a tensor

The final defining characteristic is the data type: int, float, bool, etc.

## Placeholders and variables

There are three data types for tensors in TensorFlow: Constants, placeholders, and variables.

### Constants

Immutable values.

### Placeholders

Due to the iterative approach of machine learning algorithms, placeholders are required for the input parameters to assume new values for the current iteration. For example, in the linear regression implementation, placeholders are used to take the values of the x and y coordinates for the data points for each iteration of the algorithm.

![](./markdownImages/placeholders.png)

The placeholders in this computation graph are the input nodes A and B.

### Feed dictionary

For functions in TensorFlow to have a value for the dependent variable, values for the independent variable must be declared. This can be done using a `feed_dict` which is a json object which contains the values for the independent variable used in the operation.

```python
x = tf.placeholder(tf.int32, shape=[3], name='x')

sum_x = tf.reduce_sum(x, name="sum_x")

print "sum(x): ", sess.run(sum_x, feed_dict={x: [100, 200, 300]})
```

In this simple example, x is the placeholder which is defined as an integer array of 3 values. The sum of the array is computed using `session.run()` on the `tf.reduce_sum()` on the placeholder. The feed dictionary declares the values for the placeholder in the sum operation.

In summary, the dependent variable is instantiated as a `tf.placeholder()` and given discrete values for operations via a `feed_dict()`.

### Variables

While the placeholder assumes the value of the input, a variable is declared to hold the constantly changing value of the result. Variables in TensorFlow must be first instantiated by declaring

`y = tf.Variable([1.0, 3.0], tf.float32, name='y')` and then initialized using either the global initializer:

`tf.global_variables_initializer()`

or on a specific variable using:
`tf.variables_initializer([y])` methods before they can be accessed in a TensorFlow session.

## Working with Images

In TensorFlow, working with images depends on using neural networks to perform image recognition. The pixels themselves, the fundamental building blocks of images, are converted to tensors using image recognition in the neural network algorithm.

Image recognition using neural networks is accomplished by feeding a corpus of images into a feature selection and classification algorithm, the output of which is an ML-based classifier as discussed in [neural networks](#neural-networks). This classifier can then be applied to a new image to produce a classification label. Machine learning is accomplished by first training a model, then using the corpus of images (the training data) to tweak and optimize the parameters in that model, and then you have the classifier which can be used on a new image.

## Images as Tensors

The individual pixels of an image, as described above, are converted to tensors which can be used in the TensorFlow algorithm. Each pixel holds a value based on the type of image. For grayscale images, the pixel holds a value between 0-1 to describe the level of saturation of gray in that pixel. RGB (**R**ed, **G**reen, **B**lue) is another typical form of color encoding. For RGB encoding, three values are required to describe the color in each pixel. For pure red, the numerical representation in RGB encoding would be (255, 0, 0). Likewise, blue would be (0, 0 255) and green
(0, 255, 0). These values are also called **channels** which represent the color in a pixel.

Images can also be represented in 3-D tensors. The first two elements correspond to the pixel's x-and-y coordinate location, and the third element corresponds to the number of channels of color-encoding in the image.

![](.\markdownImages\3-DRepresentation.png)

The left tensor is a grayscale image, whereas the right tensor representation is a 3-channel encoded image.

TensorFlow typically deals with 4-Dimensional shape vector representation of images, where the first value is the number of images in the list. For example, a list of 10 of the 6 pixel by 6 pixel images above with 3-channel color representation would have a shape vector of (10, 6, 6, 3) - 10 images, of 6 x 6 pixel size, and 3-channel color representation respectively.

## Coordinator
