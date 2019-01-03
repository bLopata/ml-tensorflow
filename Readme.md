# Introduction

## Main Types of machine learning

### Classification

A classification ML-algorithm is one which determines if something belongs to one or more groups.

ML-based classifiers differ from traditional "rule-based" classifiers in that:

- ML-based classifiers are dynamic while rule-based classifiers are static.
- Rule-based classifiers require experts while ML-based classifiers do not.
- ML-based classifiers require a corpus of data while rule-based classifiers do not.

**Feature vectors** are the attributes that the ML algorithm focuses on. Each data point is a list, or vector, of features. In a traditional classifier, the experts determine which features to pay attention to.

<a name="neural-networks"></a>

### Regression

Regression, or fitting a curve to a set of data points, is the simplest example of machine learning possible. Linear regression is the simplest form of regression.

The _error_ of regression is calculated by taking the sum of the squares of the residuals, or distance between the regression fit line and all of the data points.

In the LinearRegressionWithFakeData.ipynb example, the GradientDescentOptimizer library is used to solve the linear regression on a very simple data set to find the best fit line. Optimizer libraries provide simple, out of the box solutions to regression.

### Clustering

Clustering is looking at the input data and trying to find logical grouping within the data.

### Rule extraction

Determining implicit rules, or correlational relationships, within the input data.

## Neural networks

A neural network is a deep learning representation classification system. A deep learning system is one which decides it's feature vector in part or in whole independently. A deep learning-based binary classifier (seen in below image) requires a corupus of data, a feature selection & classification algorithm which contains multiple neural layers comprised of neurons. The neural layers which interact with the images directly (the Pixels and Object Parts layer which take in the image and output the image to the ML-based classifier in the example) are called visible layers, while the others are known as hidden layers.

![](./markdownImages/neuralNetwork.png)

The interaction of the individual neural layers and the neurons themselves can be seen in the following image.

![](./markdownImages/neurons.png)

The difficulty in designing a neural network is in choosing which features to map and how to architect and implement those designs.

The network contains complex interconnects between simple neurons. Different configurations lend themselves to different applications; for example a convolutional neural network would be used for image processing, while text or natural language processing applications would use a recurrent neural network. The difference between these are the interconnections between the neurons. The neurons themselves perform only two simple functions to its specific inputs: affine transformation (Wx+b) which is simply a weighted sum with a bias added, and activation function (ReLU, SoftMax, etc).

![](./markdownImages/affineTransformation.png)

The values of W and b, which are variables, are determined by TensorFlow in the training process. The objective of the training process is to determine the optimal values of W and b for each neuron. This is accomplished using the cost function, the optimizer, and run for a set number of iterations, all which are specified by the developer. During the training process, outputs from the deeper layers must be fed back to the earlier layers to determine the optimal values, this is known as back propogation. The activation function, the second step of the neuron-level operation, allows neural networks to perform linear or non-linear transformations.

# Introduction to TensorFlow

## Computation graph

A computation graph, is a directed, acyclic representation of a TensorFlow algorithm. The tensors, the arrows in the graph, flow strictly from left to right and are modified by the nodes within the graph.

![](./markdownImages/computationGraph.png)

The connections to the left of a node are known as the direct or indirect dependencies.

![](./markdownImages/computationGraphDependencies.png)

The result of the computation graph has the loss calculated, which is then fed back in to the neural network. This feedback loop can be visualized using either the neural network layered diagram:

![](./markdownImages/neuralNetworkFeedback.png)

or the computation graph via "unrolling" the graph:

![](./markdownImages/computationGraphFeedback.png)

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

## Datatypes in TensorFlow

There are three data types for tensors in TensorFlow: Constants, placeholders, and variables.

### Constants

Constants are immutable values used for storing discrete values in TensorFlow.

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

# Working with TensorFlow

## Multithreading

TensorFlow supports built-in multi-threading via the `tf.train.coordinator()` and `tf.train.start_queue_runners()` functions which handle the threads and dispatch resources as needed to complete the image rendering and manipulation.

Calling `tf.train.coordinator().request_stop()` and `tf.train.coordinator().request_stop()` will have the python interpretor wait for the tasks to complete before continuing.

## Working with Images

In TensorFlow, working with images depends on using neural networks to perform image recognition. The pixels themselves, the fundamental building blocks of images, are converted to tensors using image recognition in the neural network algorithm.

Image recognition using neural networks is accomplished by feeding a corpus of images into a feature selection and classification algorithm, the output of which is an ML-based classifier (as discussed in [neural networks](#neural-networks)). This classifier can then be applied to a new image to produce a classification label. Machine learning is accomplished by first training a model, then using the corpus of images (the training data) to tweak and optimize the parameters in that model, and then you have the classifier which can be used on a new image.

## Images as Tensors

The individual pixels of an image, as described above, are converted to tensors which can be used in the TensorFlow application. Each pixel holds a value based on the type of image. For grayscale images, the pixel holds a value between 0-1 to describe the level of saturation of gray in that pixel. RGB (**R**ed, **G**reen, **B**lue) is another typical form of color encoding. For RGB encoding, three values are required to describe the color in each pixel. For pure red, the numerical representation in RGB encoding would be (255, 0, 0). Likewise, blue would be (0, 0 255) and green
(0, 255, 0). These values are also called **channels** which represent the color in a pixel.

Images can also be represented in 3-D tensors. The first two elements correspond to the pixel's x-and-y coordinate location, and the third element corresponds to the number of channels of color-encoding in the image.

![](./markdownImages/3-DRepresentation.png)

In the above image, the left tensor is a grayscale image, whereas the right tensor representation is a 3-channel encoded image.

TensorFlow typically deals with 4-Dimensional shape vector representation of images, where the first value is the number of images in the list. For example, a list of 10 of the 6 pixel by 6 pixel images above with 3-channel color representation would have a shape vector of (10, 6, 6, 3) - 10 images, of 6 x 6 pixel size, and 3-channel color representation respectively.

### Compiling images into a list

Calling `tf.stack()` on an array of images will convert a list of 3-D tensors into a single 4-D tensor. For example, two-(224, 224, 3) tensors will become (2, 224, 224, 3) which is an array of two 224 pixel x 224 pixel, three-channel image tensors.

## Learning algorithms

A machine learning algorithm is one which is able to learn from data. "Learning" is defined as a computer program with respect to some class of tasks T, and performance measure P which improves with experience E. This performance measure could be accuracy in a classification algorithm, residual variance in regression, or a number of other metrics.

## MNIST & K-nearest-neighbor Algorithm

The Modified National Institute of Standards handwritten digit dataset, which is freely available for use [here](http://yann.lecun.com/exdb/mnist/index.html) contains 60,000 handwritten digits which we will analyze using the K-nearest-neighbor machine-learning algorithm. Each image is (28, 28, 1) and has a corresponding label containing the number in the image which can be used to optimize and improve our algorithm.

In general, there are two types of ML algorithms:

**Supervised** which uses labels associated with the training data to correct the algorithm, and **unsupervised** which requires robust set-up to learn the data as it does not use labels to self-correct.

The K-nearest-neighbor (KNN) algorithm is a supervised algorithm which uses the corpus of data to identify the closest image to the input. The algorithm accomplishes this using distance measures. Since the image as a tensor is simply a matrix of values, the distance between an input image and a training data image can be computed just like with regression. Euclidean, Hamming, and Manhattan distance measures are three examples of distance measures.

![](./markdownImages/kNearestNeighbors.png)

In the above example, the star represents the input image, and the various data points represent the training data images. The distance is computed between the input image and all data points in the training set. Based on this calculation, the algorithm will determine that the input image is a blue data point.

The L1 distance, also called the Manhattan distance, is the preferred method for dealing with distances in discrete space. This distance is found by counting the number of steps in each direction between two points.

One-hot notation is a vector which represents the value of the digit corresponding to the index of the vector. For example, a 4 would have a vector of [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] in one-hot notation, the fourth index of the vector being 1, the one-hot index, while all other indeces are zero. By definition, this notation can only be used on discrete quantities.

# Regression

## Implementing Linear Regression

Linear regression can be implemented using a simple neural network of one neuron containing a linear activation function. The implementation of an ML-based regression algorithm is as follows:

![](./markdownImages/implementingRegression.png)

An epoch is each iteration or step of the optimizer, and the batch size is the number of data points given to the optimizer for each epoch. Stochastic gradient descent optimizers use only one data point at a time, while mini-batch and batch gradient descent optimizers use a subset or the entirety of the data points, respectively, for each iteration. The goal of the optimizer is to minimize the cost function of the regression.

## Logistic Regression

Linear regression seeks to quantify effects given causes, while logistic regression seeks to quantify the probability of effects given causes. For this reason, logistic regression is also known as linear classification.

While similar, the uses of logistic regression vary, and the TensorFlow implementation is different from linear regression in two primary ways:

- logistic regression uses a softmax activation function, and
- cross-entropy as the cost function to minimize the mean-square error (MSE).

Logistic regression uses a probability function which results in an increasing probability for positive values of A and B, and a decreasing probability for negative values of A and B (per the below image)

![](./markdownImages/logisticRegression.png)

[ ] TODO: update image with no cursor.

Logistic regression requires a categorical dependent or y-variable, and can use a continuous or categorical x-variable just like linear regression. Logistic regression seeks to fit the data to an S-curve by solving for the optimal values of the A and B variables. Logistic regression can be converted to a linear form by taking the log transformation of the probability equation. This is known as the logit equation, which is defined as the natural log of the odds function.

## Implementing Logistic Regression

Logistic regression varies from linear regression implementation in two primary ways as described above:

- softmax activation function
- and cross-entropy cost function

The number of neurons required to implement logistic regression in TensorFlow is the number of classification labels required minus one. So for binary classification, our labels are True and False, and as such this requires only one neuron to implement (2 - 1 = 1).

![](./markdownImages/implementingLogisticRegression.png)

The left-hand side of the neuron-level image is identical to the linear regression neuron, with the activation function producing the probability equations as shown in the below image.

![](./markdownImages/logisticRegressionNeuralNetwork.png)

Generalizing, an M-length feature vector with N number of required classifications requires the W vector to be [M, N] and b to be [N].

![](./markdownImages/logisticRegressionGeneralized.png)

The cross-entropy cost function for logistic regression can be visualized by imagining two sets of series data: one for the actual y-values (y_actual) and one for the predicted y-values (y_predicted). Superimposing these two series on the same axis results in either the labels of the two series being lined up as in the below image, or not lined up. The lined up, or in-synch labels result in a low cross-entropy, while out of synch labels result in a high cross entropy. Cross-entropy is a way to determine if two sets of numbers have been drawn from similar or different probability distributions.

![](./markdownImages/crossEntropyVisualized.png)

### Estimators in TensorFlow

Estimators are an API in TensorFlow which provide encapsulation of training, evaluating, predicting, and exporting your TensorFlow ML-algorithm. Estimators exchange data directly with the input function, which transmits data to and from the feature vector. The estimator then handles instantiating the optimizer, fetching the training data, defining the cost function, running the optimization, and finally returning a trained model.

## Neural Networks

### Neuronal Operations

A single neuron can be classified as active if a change in the input to that neuron results in a change in the output of the neuron. If any arbitrary change to an input does not cause a change in the output, it can be said to be dead or inactive. The output of one neuron is also the input to one or many neurons in a subsequent layer. The weights on an input to a neuron define how sensitive the neuron is to that particular input. The higher the weight of an input, the more sensitive that neuron is to the input. "Neurons that fire together wire together".

A neuron, as described before, contain an affine transformation and an activation function. Examples of common activation functions and their output graphs are shown in the below image.

![](./markdownImages/activationFunctions.png)

The choice of activation function drives the design of a neural network. The combination of the affine transformation and activation function allow neural networks to learn arbitrarily complex algorithms.

Training via back propogation is a way to feed the error and output of the optimization algorithm backwards through the layers of the neural networks to tweak the weights and biases (variables W and b, respectively) in reverse sequential order to improve the accuracy of the neural network.

![](./markdownImages/backPropagation.png)

### Hyperparameters

Hyperparameters in neural networks are design decisions made by the developer to improve the performance of a neural network model. Examples of these hyperparameters are: network topology (neuron interconnections), number of neural layers, number of neurons within each layer, and the activation function used in the neuronal operation. Hyperparameters are design decisions, or inputs, used in the actual model, whereas model parameters are the weights and biases determined during the training process. Additionally, model parameters are measured using validation datasets to find the best possible model, while hyperparameter tuning is used to generate the model which is used to validate the datasets.

### Problems with Neural Networks

Neural networks are prone to several problems which cause a model to fail to be able to perform its task effectively. One is vanishing gradient, which is the term used when the loss function fails to adjust between iterations. If the gradient, or result of the loss function does not change, the iterative process to determine W and b fails to optimize correctly and will converge to an inaccurate result. The converse of this is the exploding gradient problem where the gradient moves abruptly or explodes causing diverging solutions and an inaccurate model.

These issues can be dealt with using the following methods: proper initialization of W and b using a formula, gradient clipping to establish a range of acceptable values for the gradient to prevent the value from exploding in either direction, batch normalization to limit the output of the neuron to be centered around zero, and lastly using the proper, non-saturating activation function.

Initialization of the variables should be conducted such that the variance of the outputs in each direction is equal to the variance of inputs. By initializing W to a random value for each neuron, a normal distribution is acheived. The standard deviation is calculated based on the number of inputs and outputs. Gradient clipping is most often used during the back propagation with recurrent neural networks, which are prone to exploding and vanishing gradients. Batch normalization zero-centers the inputs before passing to the activation function. The mean is then subtracted from this and that sum is divided by the standard deviation. This method is effective enough to use a saturating activation function.

Saturation occurs when the output of the activation function plateaus or is unchanging. In the logit S-curve for example, the output in the center, where the slope is non-zero, is the active or responsive region, while the left and right-hand side of the curve, where the slope is zero, is the saturation region. In these regions, the neuron might become unresponsive - that is, the output of the neuron will not change as the input changes. If this continues throughout training, and the neuron cannot be moved out of the saturation region, the neuron is said to be dead.

![](./markdownImages/saturationRegion.png)

The ReLU activation function also has a saturation region for small negative values. The ELU activation function, which has a small exponential value for the negative region, is the preferred method to dealing with the nonresponsive predisposition of the ReLU function.

### Overfitting and Underfitting

Overfitting the curve occurs when a model fits the training data too closely, but cannot accurately predict test data. This model is said to have low bias error, where few assumptions are made about the underlying data, but a high variance error, or high changes to the model with differing training data sets. An overfitted model is too complex, and too much importance is placed on the training data. The opposite is an underfitted model, which makes many assumptions about the training data, and does not change much when the training data is changed, and is too simple. Neural networks are prone to overfitting, and thus high variance. This is why it is important to test the model against a test set of data to determine the variance.

Preventing overfitting can be accomplished by _regularization_, which penalizes complex models; _cross-validation_ or having two datasets and distinct phases for training and validation; _dropout_, or intentionally turning off a subset of the neurons in the network, which causes other neurons to correspondingly adjust their output during training.

_Regularization_ is simple to accomplish with gradient descent optimizers by adding a penalty to the cost function which increases as the complexity of the model, or the function of the neuron weights, increases. This forces the optimizer to keep the model simple.

With _cross-validation_ comes hyperparameter tuning, and running training data through multiple models, choosing the model which most accurately predicts the test data.

"_Dropout_", which is deactivating a random subset of the the neurons within a NN, causes the neurons which remain on during a particular training phase to recognize patterns in the data using a different network configuration. This results in a more robust NN which is less prone to overfitting.

### Prediction Accuracy

Prediction accuracy is the primary metric for validating the efficacy of a ML-algorithm. Accuracy itself, however, is ineffective when dealing with a _skewed dataset_, that is one where certain labels are far more or far less common than other labels. Accuracy can be computed by dividing the sum of the true positive and negative results (that is, where the predicted label = actual label) divided by the total number of predictions.

Precision, which can be thought of as a measure of exactness or quality, is computed by dividing the true positive (predictied label = actual label) by the sum total number of positives (true and false).

Recall, which is the measure of completeness or quantity, can be computed by accuracy of the prediction label versus the total number of "true" labels for a binary classifier (true positive and false negative).

![](./markdownImages/confusionMatrix.png)

Stated differently,

Accuracy = (TP + TN) / (Total #)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

### Decision Threshold

The decision threshold is the point on a logistic curve at which the model predicts one label for probabilities lower than the threshold and the other label for probabilities higher than the threshold (for binary classifiers). This can also be thought of as an s-curve with Precision on the y-axis and conservativeness on the x-axis. As conservativeness increases, so too does prediction. However, a model which is too precise is also too constrained to provide accurate results; for instance, a decision threshold of 1 - or 100% - will result in only negative results. On the other hand, a model which is too liberal in predicting positive results, having too low of a decision threshold, results in an all positive result set but very low precision.

Plotting recall versus conservativeness results in a reciprocal graph to Precision versus Conservativeness graph. Thus, a model which has high recall has low precision, and a model with high precision has low recall.

![](./markdownImages/precisionRecallTradeoff.png)

### Choosing Model Characteristics

How then are we to choose a model which can maximize true results without making the negative results too common?

Two heuristics for determining the effectiveness of the model are the F1 score and the ROC.

F1 Score - the harmonic mean of precision and recall - can be computed by the following formula:

F1 = 2 X (Precision X Recall) / (Precision + Recall)

The F1 score will be closer to the lower of recall and precision and favors an even tradeoff between recall and precision. Determining the optimal values is done with an iterative process similar to hyperparameter tuning. First you tweak the threshold values before computing the precision, recall, and the F1 score of the model. The higher the F1 score, the better the model is graded.

ROC curve - or receiver operator characteristic - is a plot of the true positive rate versus the false positive rate. The ideal model has a very high TP rate, and a FP rate of zero. The ROC curve can be computed by tweaking the threshold value iteratively and computing the True Positive (TP) rate and False Positive (FP) rate, and choosing the point at the top-left of the curve.

## Convolutional Neural Networks

The convolutional neural network is ideal for processing images. A convolutional neural network is based on the visual cortex in humans. The neurons in our own brain respond to stimuli wihin it's own local receptive field, and disregard everything outside of that. These stimulated responses are then aggregated by other neurons to form patterns. For image processing in a neural network, the lower-level layers focus on larger scale details within the image, and higher-level layers focus on progressively more granular details. Within convolutional neural networks (CNNs) there are two basic types of layers: _convolution_ layers, which respond to stimuli in the local receptive field, and _pooling_, which subsamples the inputs to previous layers.

### Convolution

The most important part of a CNN are the convolution layers as the local receptive field stimuli and response are the building blocks of CNNs. Convolution can be thought of as a sliding window function applied to a matrix. For image processing, the matrix is a 2-D matrix and the function is a filter or kernel function. For a simple 6x6 matrix, we choose a 3x3 kernel as a design decision when choosing a CNN. The kernel, which can be seen in the center of the diagram below, is then overlaid with the matrix and slid sequentially from the top-left, n number of spaces left and down as chosen by the algorithm design. The sum of the unfiltered values for each step, which correlate to the local receptive field, are then represented in a 4x4 convolution matrix (right-side of the diagram).

![](./markdownImages/convolutionDiagram.png)

[The choice of the kernel function](http://aishack.in/tutorials/image-convolution-examples) depends on the application. For example, if the goal is to acheive a blurring effect, a kernel function which averages neighboring pixels would be chosen to acheive that effect. Kernel functions can be designed to acheive many complex image effects, such as edge and line detection.

### Design Features of CNNs

_Zero padding_, adding a certain number of rows and columns of zeroes to the edges of your data, is used in order to have every element represented in the filtered result. Without zero padding, some elements will not be represented in the convolution matrix. Zero padding can also be used to acheive _wide-convolution_ which is when a convolution matrix is larger than the input matrix.

_Stride size_ is the number of rows to skip when moving horizontally and columns to move when moving vertically when sliding the kernel function. This determines how much of the local receptive field which will overlap when performing convolution. The lower the stride size, the greater the overlap.

### CNNs versus DNNs

Dense neural networks, those which have an interconnection between each neuron of one layer with all neurons of a previous layer, have far too much complexity to be used effectively in image processing. For a 100 pixel by 100 pixel image, you would need 10,000 neurons in the first layer. With a dense neural network, this results in millions of interconnections by the second layer - millions of parameters to tune during training! CNNs, by way of the use of local receptive fields, have dramatically fewer parameters versus DNNs since they focus on only a small area within the image rather than trying to determine patters for every part of the image at once. CNNs also identify patterns independent of location whereas DNNs will inherently parse location information as well as image data due to their architecture.

### Feature Maps

Convolutional layers are comprised of feature maps, which are themselves comprised of a number of neurons, each having received values based on the values of the local receptive field of the kernel function from the previous layer. Because the feature map is created from the same kernel function, all neurons within each feature map have the same weights and biases. CNNs are sparse neural networks since there is not a 1:1 correlation between neurons in two adjacent layers.

Each neuron's receptive field includes all the feature maps of all previous layers. In this way, aggregated features are processed in convolutional layers. For a visualization of how CNNs are constructed, see the image below. Many feature maps comprise a convolutional layer, and many convolutional (and pooling layers) comprise a single CNN.

![](./markdownImages/explodedViewCNN.png)

### Pooling Layers

Pooling layers subsample inputs into convolution layers. The neurons in a pooling layer have no associated weights or biases. A pooling layer neuron simply applies an aggregate function to all inputs. Pooling layers greatly reduce time and memory usage during training by reducing the numbers of parameters via aggregation and also mitigate overfitting to test data via subsampling. Pooling layers also allow NN to recognize features indpendent of location. Pooling is typically done on each channel independently.

### CNN Architectures

CNNs are typically comprised of alternating convolutional and pooling layers. The output of each of the convolutional and pooling layers is an image, and the images shrink in size successively due to the subsampling done in the pooling layers. Each successive output image is also deeper due to the feature maps in the convolutional layer. The output of the entire set of these convolutional and pooling layers is then fed into a regular, feed-forward dense neural network which has a few, fully-connected layers each with a ReLU activation function and finally a SoftMax prediction layer to provide classification. For digit classification, there are 10 prediction labels, for image classification there can be more or fewer prediction labels.

## Recurrent Neural Networks

### Recurrent Neurons

Recurrent Neural Networks (RNNs) are based upon a recurrent neuron, that is a neuron which has memory or state. Unlike normal neural networks or convolutional neural networks the output of a recurrent neuron is fed back in as an input to the same neuron. This feedback makes RNNs well-suited for time series data. RNNs are known as auto-regressive because the output at time `t` is dependent on the output at time `t-1`.

![](./markdownImages/recurrentNeuron.png)

### Recurrent vs Normal Neuron

For a regular neuron, the input is a vector which produces a scalar output. However, a recurrent neuron with an input feature vector of [X<sub>0</sub>, X<sub>1</sub>, ..., X<sub>t</sub>] would produce an output vector of [Y<sub>0</sub>, Y<sub>1</sub>, ..., Y<sub>t</sub>]. Additionally, while a regular neuron has one weight vector, a recurrent neuron will have two: W<sub>y</sub> for the previous y-input, and W<sub>x</sub> for the indpendent x-input.

As recurrent neurons primarily deal with time-series data, it can be useful to think about the neuron for each instance of time for a given set. One way to visualize this is through "unrolling" the recurrent neuron through time. That is, showing the neurons inputs and outputs as plotted along a time axis. In the below image, notice how the output of the neuron at t=0 feeds into the input of the same neuron at t=1.

![](./markdownImages/unrollingRNN.png)

A layer within an RNN is generally a group of recurrent neurons, known as a RNN or memory cell. The same process for unrolling through time is performed on this memory cell for as many time instances as there are datapoints.

### Training an RNN

Gradient descent optimizers, which seek to minimize the mean square error (MSE) for values of W and b, are used in RNNs as well. However, training of RNNs is accomplished through back-propagation through time (BPTT). BPTT is very similar to back-propagation, however BPTT has a few more details to consider as we unroll the RNN through time. The number of layers needed for an RNN depends on the number of time periods you wish to study. Because RNNs can be unrolled very far back in time, RNNs which rely upon time periods in the very distant past are especially prone to vanishing and exploding gradients as the gradient needs to be propagated back through each time instance.

One option to mitigate vanishing and exploding gradients in RNNs is to use truncated BPTT. Truncated BPTT uses only a subset of data points for time periods in the very distant past which can reduce the accuracy of the model. Another option is to use long short-term memory (LTSM) cells. LTSM cells were developed specifically to deal with vanishing and exploding gradients.

### Long Memory Neurons

In order to combat the problems with vanishing and exploding gradients in deep recurrent neural networks, the state of a memory cell must be expanded to include long-term state.

# Labs

## Logistic Regression

For the logistic regression lab, we are utilizing the single neuron implementation of logistic regression in TensorFlow to determine the probability of Google stock having an increasing or decreasing return from one month to the neYt by classifying the returns of the S&P 500 index as our independent variable. We have used pandas and numpy in determining the baseline, and will compare that result to the ML-based logistic regression.

The softmax activation function is invoked using the following method in the `tf.nn` TensorFlow neural network library:

```python
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

`y_` is our placeholder for the output label of the softmax function (either True - 1 or False - 0) in One Hot notation. `y` is the output of the affine transformation, or the input to the softmax activation function `y=W*x + b`.

We use `tf.reduce_mean` on this softmax activation function to compute our cross-entropy calculation to compare the probability distributions of our predicted value against the actual value (Google stock increasing/decreasing versus S&P increasing/decreasing).

## MNIST & KNN

For this lab we are using the MNIST dataset of hand-written numbers. In our comparison using the KNN algorithm, the 28 pixel x 28 pixel image tensors are flattened into vectors of length 784, which are then compared against the training data by summing the training data vector and the negation of the test data in 5,000-element chunks. This results in 5,000 vectors of length 784 containing the L1 distance between each pixel in the test image against the MNIST training data chunk. The sum of the absolute value of these vectors is then computed and reduced to a single-element vector per distance vector using `tf.abs()` and `tf.reduce_sum()`.

For K = 1, `numpy.argmin()` can be used to find the single nearest neighbor for our test image, then the label of the nearest neighbor from the training data can be used to compare against the test digit to perform the supervised optimization of the algorithm.

## Neural Network Automobile Prices

For this lab we used a public dataset of automobiles from UCI as the training data to an ML-based predictor of automobile price given various categorical and numerical features from the dataset such as make, engine type, miles-per-gallon, etc. We created a pandas dataframe to read in and clean up the data and passed it into TensorFlow using `tf.estimator.inputs.pandas_input_fn()` which is a built-in method in TensorFlow which takes in a pandas data frame as an input. We defined `feature_columns` as an array of both categorical and numerical column data as unique entries for each column in the dataset. Scaling the price column to tens of thousands of dollars rather than the full price was used to improve accuracy as TensorFlow works better with smaller numbers. These scaled values were converted back to dollars after the training.

We tweaked the neural network configuration using the `hidden_units` parameter of the `tf.estimator.DNNRegressor()` method between a two-layer and a three-layer configuration to demonstrate the effectiveness of each on our resulting training model. The accuracy improved substantially when using a three-layer DNN (dense neural network) with the configuration of [24, 16, 24] neurons rather than the two-layer configuration of [20, 20] neurons.

## Iris flower DNN Classifier

For this lab, we are working with the iris data set. The objective of this ML model is to predict the label of the iris based on the features which are the Sepal length and width and petal length and width. Rather than using pandas for this lab, we are using TensorFlow to iterate over the .csv dataset by invoking `tf.decode_csv()` which extracts the header data from the .csv file. The features are created by zipping the feature names into a dictionary for each line in the iterator. We invoke `tf.data.TextLineDataset().map()` in our helper method `get_features_labels(filename, shuffle=False, repeat_count=1)` which allows for shuffling to randomize the order of the data, `repeat_count` allows for copying of the dataset, and we specify the `batch_size` as 32. We use the `dataset.make_one_shot_iterator()` method which iterates over the dataset exactly once.

# Convolution Neural Network

For this lab we are using the publicly available house number dataset from Stanford. The house numbers are in a matlab file format, which requires additional python libraries `scipy` and `scipy.io` to read in the files and `matplotlib` and `matplotlib.pyplot` which allows for inline plotting in IPythonNotebooks in addition to the usual numpy, pandas, and tensorflow libraries. Similar to the MNIST number lab, the goal is to create an ML-based classifier which can predict the number represented in an image. However, the shape of our image tensor is now (32, 32, 3) which is a larger, color image than the MNIST dataset.

Our CNN is specified to have two convolutional layers and one pooling layer. The first convolutional layer, `conv1` is defined as having 32 feature maps, with a kernel size of 3x3, a stride of 1, with a padding of `"SAME"`, which means that the size of the output is equal to the size of the input divided by the stride size and the input will be zero-padded as necessary to acheive this result. The second layer, `conv2` is defined as having 64 feature maps, a 3x3 kernel size, and a stride of 2, also with `"SAME"` padding. The single pooling layer has the same number of feature maps as the final convolutional layer, `conv2` which is 64 - we store this in a variable `pool3_feature_maps`.

The DNN is a single, feed-forward layer, `n_fullyconn1`, which has 64 neurons and 11 outputs to accomodate label values between 1-10.

We call `tf.reset_default_graph()` to clear any nodes that have been added to the default x. The `x` placeholder input is defined as `X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")` and the y is similiarly defined as an array of integers with shape [None]. We declare the first convolutional layer, `conv1` as follows:

```python
conv1 = tf.layers.condv2d(X, filters=conv1_feature_maps),
kernel_size=conv1_kernel_size,
strides=conv2_stride, padding=conv1_pad,
activaiton=tf.nn.relu, name="conv1")
```

`conv2` is similarly declared. The shape of these two convolutional layers is as follows:

```python
conv1.shape
TensorShape([Dimension(None), Dimension(32), Dimension(32), Dimension(32)])

conv2.shape
TensorShape([Dimension(None), Dimension(16), Dimension(16), Dimension(64)])
```

The output of the `conv2` is 16 by 16 because we chose a stride length of two.

The pooling layer is instantiated using `tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")` which means the input comes from `conv2`. ksize, which is an array of parameters defined by `[batch_size, height, width, channels]`

TensorFlow does support pooling across multiple channels in an image, **or** pooling across the height and the width, but not both.

We similarly specify `strides=[batch_size, height, width, channels]` and `"VALID"` padding, which means the input will not be zero-padded.

```python
pool3.shape
TensorShape([Dimension(None), Dimension(8), Dimension(8), Dimension(64)])
```

We flatten the output of the pooling layer into a single 1-D array using `tf.reshape(pool3, shape=[-1, pool3_feature_maps * 8 * 8])` and feed the output of the pooling layer into a DNN which we label as `logits` and is defined by

```python
tf.layers.dense(pool3_flat, n_fullyconn1, activation=tf.nn.relu, name="fc1")
```

We set up our cost function using the cross entropy as we did in the logistic regression lab. We pass in the DNN defined in the above line to

```python
tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
```

Once again, we calculate the lost by calculating `tf.reduce_mean()` on this cost function, the optimizer chosen for this lab is the Adam optimizer which we call with `tf.train.AdamOptimizer()` and have this minimize the loss. We define an accurate prediction as `tf.nn.in_top_k(logits, y, 1)` to determine if the predicted result is equal to the training label, and calculate the accuracy with `tf.reduce_mean()` on the correctly predicted labels.

We define a helper method to determine the start and end indices for our training data set.

```python
current_iteration = 0

def get_next_batch(batch_size):

    global current_iteration

    start_index = (current_iteration * batch_size) % len(y_train)
    end_index = start_index + batch_size

    x_batch = x_train[start_index: end_index]
    y_batch = y_train[start_index: end_index]

    current_iteration = current_iteration + 1

    return x_batch, y_batch
```

The parameters chosen are for a batch size of 10,000 data points with 10 epochs or iterations. We instantiate a TensorFlow using `tf.Session()` and initialize our variables using `tf.global_variables_initializer.run()`. The accuracy is computed within each epoch to show the improvement of the model as the optimizer is run.

```python
with tf.Session() as sess:
    init.run()

    num_examples = len(y_train)
    for epoch in range(n_epochs):
        for iteration in range(num_examples // batch_size):

            X_batch, y_batch = get_next_batch(batch_size)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        acc_test = accuracy.eval(feed_dict={X: x_test, y: y_test})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
```

The output for each epoch is as follows:

0 Train accuracy: 0.48 Test accuracy: 0.45933333

1 Train accuracy: 0.68 Test accuracy: 0.606

2 Train accuracy: 0.7 Test accuracy: 0.696

3 Train accuracy: 0.77 Test accuracy: 0.718

4 Train accuracy: 0.8 Test accuracy: 0.738

5 Train accuracy: 0.83 Test accuracy: 0.74733335

6 Train accuracy: 0.86 Test accuracy: 0.746

7 Train accuracy: 0.89 Test accuracy: 0.77066666

8 Train accuracy: 0.9 Test accuracy: 0.758

9 Train accuracy: 0.87 Test accuracy: 0.7673333

## Building a CNN Using Estimator API

In this lab we are returning to the MNIST dataset, however this time we will be constructing a custom convolutional neural network (CNN) using TensorFlow's Estimator API. The MNIST dataset contains images of a single digit which are 28 pixels x 28 pixels in grayscale, or a (28, 28, 1) image tensor. Once again in this lab, we set up the needed import statements and pull the dataset directly from Google. We define variables for the height, width, and channels of the images, and also the various features of each convolutional layer wihin the CNN: the number of feature maps, the kernel and stride sizes, and zero padding.

We then define a helper function to build a custom CNN for a defined set of features. This helper function first reshapes the input data into the appropriate dimensions for the images. Next we define two convolutional layers and one pooling layer using `tf.layers.conv2d()` and `tf.nn.max_pool()` and pass in the previously declared variables for the CNN. We are using the ReLu activation function for this CNN. The pooling layer is then flattened into a 1-D array to be connected to the dense neural network (DNN) which has 64 neurons. The return value of this helper function is the output of the logits layer which will later be optimized using cross-entropy as performed in the previous lab.

In order TensorFlow's estimator API, we need to set up a model function which has three inputs: "features", "labels", and "mode". The "mode" being either prediction, evalutation, or training. The "features" are a dictionary of specifications which we pass in to the helper function we created in the previous step to build our CNN. For the first mode, prediction, all we need as an output is the output of the softmax activation function.

```python
logits = build_cnn(features)

predicted_classes = tf.argmax(logits, axis=1)
```

Where `build_cnn()` constructs the CNN as fefined previously. If the purpose of this CNN, or it's mode, is prediction, then we simply return these predicted classes, i.e.

```python
if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode, predictions=predicted_classes)
```

Otherwise we will calculate the loss, again using `tf.reduce_mean()` and setup the optimizer, once again we are using the Adam optimizer, and minimize the loss as done in the previous CNN lab. We set our optimizer to run for 2000 steps with a batch size of 100 and print out the loss for every 100 steps. After running a test model, we obtain an accuracy of 98.6% - far higher than the previous CNN lab due to the decreased complexity of the dataset.

## RNNs for Image Classification

For this lab we are using the MNIST dataset using recurrent neural networks (RNNs). We begin by importing the modules for the lab: TensorFlow, matplotlib, and numpy as well as the MNIST dataset downloaded directly from the TensorFlow example tutorial. Next we reshape the images to a 28 pixel x 28 pixel grayscale image using

```python
def display_digit(digit):
    plt.imshow(digit.reshape(28, 28), cmap="Greys", interpolation='nearest')
```

We then define the RNN as having 28 time-instances which is also the number of layers, 200 neurons per memory cell (tuneable hyperparameter), 10 outputs for the digits 0-9, and 28 inputs to each step, equal to the width of each row in our image. We structure the image array such that every row is one step in time.

We then instantiate placeholders for the x and y-values of our image tensor. Next we set up the RNN. TensorFlow abstracts much of the RNN creation away from the user. Since we are working with relatively simple images, we set up a basic memory cell using

```python
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=28)
```

And we store the output and state of the previous memory cell using

```python
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
```

Since this is an image classification problem, we then set up the `logits` layer, a dense layer having 10 outputs, each a probability for the corresponding digit, and the input to which is the final state of the RNN. We then set up the cross-entropy and loss calculations, and the optimizer which is once again the Adam optimizer. The output of the cross-entropy is the predicted label with the highest probability.

```python
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

training_op = optimizer.minimize(loss)
```

We initialize all of our variables and reshape the test digits using the helper method declared above. For our TensorFlow session, we set up 10 epochs of 150 images per iteration. We set up the feed dictionaries to pass all x and y-values into the optimizer, and the training and test accuracy is computed at each epoch. For this simple dataset, we were able to acheive training accuracy of >97% and test accuracy of >95%.
