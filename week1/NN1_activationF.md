# Activation Functions

As we saw in the previous section, the nodes in hidden layers, aka the "activation units," receive input values from data or activation units of the previous layer. Each time a weighted sum is computed. Then the activation function defines which value, by consequence importance, the node will output. Before going of the most common activation functions for neural networks, it is essential first to introduce their properties as they illustrate core concepts or neural network learning process.

## Mathematical properties and consequences

### Differentiability
We will see in the next lecture that the backpropagation, the algorithm adjusting all network's weights and biases, involves a gradient descent procedure. It is thus desirable for the activation function to be continously differentiable (but not strictly necessary, as we will see soon for particular functions). The Heaviside step function of the perceptron has a derivative undefined at $z=0$ and the gradient is zero for all $z$ otherwise: a gradient descent procedure will not work here as it will 'stagnate' and never start descending as it always returns zero.

### Range
The range concerns the interval of the activation function's output values. In logistic regression, we introduced the sigmoid function mapping the entire input range $z \in \mathbb{R}$ to the range [0,1], ideally for binary classification. Activation functions with a finite range tend to exhibit more stability in gradient descent procedures. However it can lead to issues know as Vanishing Gradients explained in the next subsection {ref}`NN1:activationF:risksGradient`.

### Non-linearity
This is essential for the neural network to __learn__. Explanations. Let's assume there is no activation function. Every neuron will only be performing a linear transformation on the inputs using the weights and biases. In other words, they will not do anything fancier than $(\sum wx + b)$. As the composition of two linear functions is a linear function itself (a line plus a line is a line), no matter how many nodes or layers there are, the resulting network would be equivalent to a linear regression model. The same simple output achieved by a single perceptron. Impossible for such an network to learn complex data patterns.  

What if we use the trivial identify function $f(z) = z$ on the weighted sum? Same issue: all layers of the neural network will collapse into one, the last layer will still be a linear function of the first layer. Or to put it differently: it is not possible to use gradient descent as the derivative of the identity function is a constant and has no relation to its input $z$. 

There is a powerful result stating that only a three-layer neural network (input, hidden and output) equiped with non-linear activation function can be a universal function approximator within a specific range:

````{prf:theorem}
:label: unitheodef
In the mathematical theory of artificial neural networks, the _Universal Approximation Theorem_ states that a forward propagation network of a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $\mathbb{R}^n$.
````
When is meant behind "compact subsets of $\mathbb{R}$ is that the function should not have jumps nor large gaps. This is quite a remarkable result. The simple multilayer perceptron (MLP) can thus mimick any known function, from cosine, to exponential and even more complex curves!

## Main activation functions
Let's present some common non-linear activation functions, their characteristics, with the pros and cons.  

### The sigmoid function
Defined as:
```{math}
\sigma(z) = \frac{1}{1 + e^{-z}}
```
__Pros__  
* It is a very popular choice, mostly due to the output range from 0 to 1, convenient to generate probabilities as output.   
* The function is differentiable and the gradient is smooth, i.e. no jumps in the ouput values.

__Cons__  
* The sigmoid's derivative vanishes at its extreme input values ($z \rightarrow - \infty$ and $z \rightarrow + \infty$) and is thus proned to the issue called _Vanishing Gradient_ problem (see {ref}`NN1:activationF:risksGradient`).

### Hyperbolic Tangent
Alike the sigmoid, the hyperbolic tangent is S-shaped and continously differentiable. However the output values range from -1 to 1. 
```{math}
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
```

__Pros__  
* It is zero-centered. Unlike the sigmoid when we had to have a decision boundary of $0.5$ (half the output range), here the mapping is more straightforward: negative input values gets negative output, and positive input values will be positive, with one point ($z=0$) returning a neutral output of zero.
* That fact the mean of the ouput values is close to zero (middle of the output range) makes the learning easier.

__Cons__  
* The gradient is much steeper than for the sigmoid (risk of jumps while descending)
* There is also a _Vanishing Gradient_ problem due to the derivative cancelling for $z \rightarrow - \infty$ and $z \rightarrow + \infty$.  

### Rectified Linear Unit (ReLU)


### Leaky ReLU


### Parametric ReLU

### Exponential Linear Units (ELUs) 

### Softmax Function
It is a combination of multiple sigmoids. 

### Swish
It is a sigmoid multiplied with the identy:

### Gaussian Error Linear Unit (GELU)


## How to choose the right activation function

(NN1:activationF:risksGradient)=
### The risk of vanishing or exploding gradients



_All hidden layers usually use the same activation function. However, the output layer will typically use a different activation function from the hidden layers. The choice depends on the goal or type of prediction made by the model._



_https://datascience.stackexchange.com/questions/27665/what-is-saturating-gradient-problem_


_Mostly, Neural Networks go for different variations of RELU for its simplicity and easy computation both during forward and backward. But, in certain Cases, Other Activation Functions give us better results, Like Sigmoid is used at the end layer when we want our outputs to be squashed between [0,1], or tanh is being used in RNNs and LSTMs._ 