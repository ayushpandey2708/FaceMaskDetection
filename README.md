# FaceMaskDetection

## Bias/Variance
If  model is underfitting  it has a "high bias"
If  model is overfitting then it has a "high variance"
Model will be alright if we balance the Bias / Variance
we usually use regularization to reduce overfitting(variance) by converting some of the weights to 0

## mini_batch gradient descent
Training NN with a large data is slow. So to work on mini batches speed up our processing.
It has to be a power of 2 (because of the way computer memory is layed out and accessed)
Mini-batch size is a hyperparameter

## Adam Optimization Algorithm
It is known as adaptive moment estimation
Adam optimization simply puts RMSprop and stochastic gradient descent with momentum together
It helps the cost function to go to the minimum point in a more fast and consistent way.
It increases learning rate and used for bias correction
	vdW = (beta1 * vdW) + (1 - beta1) * dW     # momentum
	vdb = (beta1 * vdb) + (1 - beta1) * db     # momentum
			
	sdW = (beta2 * sdW) + (1 - beta2) * dW^2   # RMSprop
	sdb = (beta2 * sdb) + (1 - beta2) * db^2   # RMSprop

## Parameters vs Hyperparameters
Main parameters of the NN is W and b
Hyper parameters are the parameters that control the algorithm:
Learning rate.
Number of iteration.
Number of hidden layers L.
Number of hidden units n.
Choice of activation functions.
In the earlier days of DL and ML learning rate was often called a parameter,
but it really is and now everybody call it a hyperparameter.

## Gradient Descent
Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function.
Gradient descent is simply used to find the values of a function's parameters (coefficients) that minimize
a cost function as far as possible.
θ:=θ− α/m(X^T(Xθ− y)  (θ= parameter, m=no.of examples X=image y=whether mask(1) or not mask(0) ,α=learning rate)
Gradient is the slope of the function 
The higher the gradient, the steeper the slope and the faster a model can learn. But if the slope is zero, 
the model stops learning.

A good way to make sure gradient descent runs properly is by plotting the cost function as the optimization runs.
Put the number of iterations on the x-axis and the value of the cost-function on the y-axis.
This helps to spot how appropriate our learning rate is. 
the image illustrates the difference between good and bad learning rates.

https://builtin.com/sites/default/files/styles/ckeditor_optimize/public/inline-images/gradient-descent-plot.png


## Learning Rate
How big the steps are gradient descent takes into the direction of the local minimum are determined by the learning rate,
which figures out how fast or slow we will move towards the optimal weights.

For gradient descent to reach the local minimum we must set the learning rate to an appropriate value,
which is neither too low nor too high. This is important because if the steps it takes are too big,
it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent.
If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but that may take a while. 

https://builtin.com/sites/default/files/styles/ckeditor_optimize/public/inline-images/gradient-descent-learning-rate.png



## Activation functions
### Sigmoid
Sigmoid can lead us to gradient decent problem where the updates are so low.
Sigmoid activation function range is [0,1] A = 1 / (1 + np.exp(-z)) (Where z is the input matrix).
### Tanh
Tanh activation function range is [-1,1] .
It turns out that the tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero,
and so it centers the data better for the next layer.
Sigmoid or Tanh function disadvantage is that if the input is too small or too high, the slope will be near zero which will cause us the gradient decent problem
### Relu
One of the popular activation functions that solved the slow gradient decent is the RELU function. RELU = max(0,z) 
so if z is negative the slope is 0 and if z is positive the slope remains line
So if classification is between 0 and 1, use the output activation as sigmoid and the others as RELU.
### Softmax
Softmax is a very interesting activation function because it not only maps our output to a [0,1] range but also maps each output in such a way that the total sum is 1.
The output of Softmax is therefore a probability distribution.
The softmax function is often used in the final layer of a neural network-based classifier.
In conclusion, Softmax is used for multi-classification in logistic regression model whereas Sigmoid is used for binary classification in logistic regression model
![alt text](https://miro.medium.com/max/353/1*OC_DUasjBVuWJcbIhUcd4g.png)

![alt text](https://miro.medium.com/max/3000/1*ZeythBexsRfY97H0D0LETA.png)



## Data Augmentation

If data is increased, deep NN performs better. Data augmentation is one of the techniques that deep learning uses to increase the performance of deep NN.
Some data augmentation methods that are used for computer vision tasks includes:
Mirroring.
Random cropping.
The issue with this technique is that we might take a wrong crop,so we should make our crop big enough.
Rotation.
Shearing.
Local warping.
colour shiffting

## Pooling layers
We use pooling layers to reduce the size of the inputs and speed up computation.
It shrinks output and reduce size of the inputs.
It works well in practice and reduce computations.
Average pooling takes averages of values.

![alt text](https://github.com/mbadry1/DeepLearning.ai-Summary/raw/master/4-%20Convolutional%20Neural%20Networks/Images/02.png)


## Dropout Regularization
The dropout regularization eliminates some neurons/weights on each iteration based on a probability
At test time we don't use dropout because at test time it would add noise to predictions.
A lot of researchers use dropout with Computer Vision because they have a very big input size and almost never have enough data,
so overfitting is the usual problem. And dropout is a regularization technique to prevent overfitting.

## Binary Cross-Entropy
we will use binary cross-entropy here because it is a sigmoid activation plus a cross entroppy loss
binary cross entroppy is independent for each class means the loss computed by every CNN vector component will
not affected by other component values.

## MobileNet
MobileNet are very faster in process than cnn 
It also uses lesser parameters
It also has some disadvantages it tends to be less accurate in comparison to its competetors
But for the face recognition it shares the equal accuracy with its competetors.
We will prefer imagenet as a pretained weights too.
