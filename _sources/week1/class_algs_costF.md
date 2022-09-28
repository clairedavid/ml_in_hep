# Cost Function for Classification


## Wavy least squares
If we plug our sigmoid hypothesis function $h_\theta(x)$ into the cost function defined for linear regression (Equation {eq}`costFunctionLinReg` from Lecture 2), we will have a complex non-linear function that could be non-convex. The cost function could take this form: 

```{glue:figure} poly3minima_example
:name: "poly3minima_example"
:figwidth: 80%
```

Imagine running gradient descent starting from a randomly initialized $\theta_0$ parameter around zero (or worse, lower than -2). It will fall into a local minima. Our cost function will not be at the global minimum! It is crucial to work with a cost function accepting one unique minimum.


## Building a new cost function
As we saw in the previous section, the sigmoid fits the 1D data distribution very well. Our cost function will use the hypothesis $h_\theta(x)$ function as input. Recall that the hypothesis $h_\theta(x)$ is bounded between 0 and 1. What we need is a cost function producing high values if we mis-classify events and values close to zero if we correctly label the data. Let's examine what we want for the two cases:

__Case of a signal event:__  
A data point labelled signal verifies by our convention $y=1$. If our hypothesis $h_\theta(x)$ is also 1, then we have a good prediction. The cost value should be zero. If however our signal sample has a wrong prediction $h_\theta(x) = 0$, then the cost function should take large values to penalize this bad prediction. We need thus a strictly decreasing function, starting with high values and cancelling at the coordinate (1, 0). 

__Case of a background event:__  
The sigmoid can be interpreted as a probability for a sample being signal or not (but note it is not a probability distribution function). As we have only two outcomes, the probability for a data point to be non signal will be in the form of $1 - h_\theta(x)$. We want to find a function with this time a zero cost if the prediction $h_\theta(x) = 0$ and a high cost for an erroneous prediction $h_\theta(x) = 1$.

Now let's have a look at these two functions:

```{glue:figure} log_h_x
:name: "log_h_x"
:figwidth: 100%
```

For each case, the cost function has only one minimum and harshly penalizes wrong prediction by blowing up at infinity.  
How to combine these two into one cost function for logistic regression?  
Like this:

````{prf:definition}
:label: costFLogRegDef
The __cost function for logistic regression__ is a defined as:
```{math}
:label: costFunctionLogReg
J(\theta) = - \frac{1}{m} \sum^m_{i=1} \left[ \;\; {\color{RoyalBlue}y^{(i)} \log( h_\theta(x^{(i)} )) }\;\;+\;\; {\color{OliveGreen}(1- y^{(i)}) \log( 1 - h_\theta(x^{(i)} ))} \;\;\right]
 ```
 This function is also called __cross-entropy loss function__ and is the standard cost function for binary classifiers.
````

Note the negative sign factorized at the beginning of the equation. Multiplying by ${\color{RoyalBlue}y^{(i)}}$ and ${\color{OliveGreen}(1 - y^{(i)})}$ the first and second term of the sum respectively acts as a "switch" between the cases ${\color{RoyalBlue}y=1}$ and ${\color{OliveGreen}y=0}$. If $y=0$, the first term cancels out and the cost takes the value of the second. If $y=0$, the second term vanishes. The two cases are combined into one mathematical expression.

## Gradient descent
The gradient descent for classification follows the same procedure as described in Algorithm {prf:ref}`GD_algo_multi` in Section {ref}`warmup:linRegMulti:graddesc` with the definition of the cost function from Equation {eq}`costFunctionLogReg` above.

### Derivatives in the linear case
````{margin}
Recall $h_\theta (x^{(i)}) =  f(\theta^{T} x^{(i)})$ 
````
Consider the linear assumption $\theta^{\; T} x^{(i)} = \theta_0 + \theta_1 x_1 +  \cdots  + \theta_n x_n$ as input to the sigmoid function $f$. 
The cost function derivatives will take the form:

```{math}
:label: costfderivlin
\frac{\partial}{\partial \theta_j} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) -  y^{(i)}\right) x_j^{(i)}
```
This takes the same form as the derivatives the linear regression (Equation {eq}`partialDevLinReg` in Section {ref}`warmup:linregmulti:graddesc`).

```{admonition} Exercise
:class: seealso
To convince yourself, derive Equation {eq}`costfderivlin` starting from the general definition in Equation {eq}`costFunctionLogReg`.

Hints and help available on demand after class.
```

### Alternative techniques (advanced)
Numerous methods have been developed to find optimized $\theta$ parameters in faster ways than the gradient descent. These are beyond the scope of this course and usually available as libraries within python (or other languages). Below is a list of the most popular ones:

```{admonition} Reading
:class: seealso
* [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) and [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
* [Conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
```

## Multiclass 
We treated the binary classification problem. How to adapt to a situation with more than two classes?  

We cut the problem in a collection of binary classifiers for each class. For instance with five classes labelled A, B, C, D and E, the exercise is a binary classification problem for each:

* 
Resume this: 
https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/