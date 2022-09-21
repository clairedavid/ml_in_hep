# Linear Regression using Gradient Descent

## Model representation
We will study the following situation where we want to predict a real-valued output $y$ based on a collection of input values $x$ that would be spread in the following way:

```{glue:figure} plot_linReg_50pts
:figwidth: 80%
:name: "plot_linReg_50pts"
```
Now I see what you are thinking: it's very straightforward (pun intended), it is just about fitting a straight line to the data. Yes. But this over-simplified setup is the starting point of our machine learning journey as it contains the basic mathematical machinery. Things will complicate soon, don't worry.  

So, what is linear regression to start with?  

````{prf:definition}
:label: linRegDef
Linear regression is a model assuming a linear relationship between input variables and real-valued ouput variables.

* Input variables are called _independent variables_, or _explanatory variables_.

* The output variable is considered a _dependent_ variable.

Linear regression is used to __predict a real-valued ouput variable__ (dependent variable) based on the values of the input variables (independent variables).
````

````{prf:definition}
:label: simpleRegDef
* If there is one explanatory variable, this is __simple linear regression__ or __univariate linear regression__.

* In the case of several explanatory variables, it is called __multiple linear regression__.
````

Let's now introduce terms more specific to the machine learning jargon and define some notations we will use through the course.

```{admonition} Terminology and Notation 
* The input variables are called __features__ and are denoted with $x$.
* The ouput variable is the __target__ and is denoted with $y$.

* In supervised learning the dataset is called a __training set__.
* The number of training examples is denoted with $m$.
* The $i^{th}$ example is $(x^{(i)} , y^{(i)})$.
```

So the pair $(x^{(0)} , y^{(0)})$ is the first training example from the data set, and $(x^{(m)} , y^{(m)})$ is the last.

```{warning}
Here we start counting from one. When you will write code, the convention is to start at index zero. So your last sample will be of index `m - 1`. Keep this in mind.
```

When we refer to the entire list of all features and targets, we often use the upper letter, $X$ and $Y$ respectively. Those are __vectors__. 

We defined the input and ouput. In the middle is our model. We feed it first with all the input features and their associated known targets. 
This first step of supervised learning is called the __training__ and we will see the mathematics behind it now. What we need first is a function that best maps input to output.

````{prf:definition}
:label: hypothesisFunction
The hypothesis function, denoted $h$, is a mapping function used to predict an output $y$ from an input $x$:  
\begin{equation*}
y = h(x)
\end{equation*}
````

In our simple case of linear regression, our function $h$ will be of the form:
```{math}
:label: h_theta_lin
  h_\theta(x) = \theta_0 + \theta_1 \; x
```

The subscript $\theta$ means that the function depends on the values taken by $\theta_0$ and $\theta_1$.

````{prf:definition}
:label: modelParameters
The mapping function's internal variables are called the model parameters. They are denoted by the __vector__ $\Theta$:
\begin{equation*}
\Theta  = \begin{pmatrix} 
\theta_0 \\
\theta_1 \\
 \\
... \\ 
 \\
\theta_n \\
\end{pmatrix}
\end{equation*}
````

This is written in a general form of a polynomial of $n^\text{th}$ order. In our case with the linear regression, we only need two parameters:
```{math}
:label: theta_0_1
\Theta = \begin{pmatrix} 
\theta_0 \\
\theta_1 \\
\end{pmatrix}
```

We want to find the values of $\theta_0$ and $\theta_1$ that fit the data well.
We could pick one training example $(x^{(k)} , y^{(k)})$ and derive the coefficients from there. But will this be the 'best' straight line to draw?
The mathematical phrasing for such a task is to think in terms of errors. How do we calculate the errors? That's a first question to ask. 
From a given vector of $\Theta$, how small are the errors?
This picture below helps to visualize. From a given parameterization, that is to say a given tuple ($\theta_0$ , $\theta_1$ ), the mapping function will ouput continuous values of a predicted $y$ for a continuous range of $x$. That is the dashed line. The errors are the (vertical) intervals between the $y$ from the prediction and each data points. 
```{figure} ../images/lec02_1_square_err_graph.png
---
width: 60%
name: squareErrVisual
---
. Visualization of errors (dotted vertical lines) between observed and predicted values.  
Image: Don Cowan.
```
To see how well the prediction fit the data, we want the sum of all these errors to be as small as possible. In other words, we want to solve a minimization problem. 

To avoid cancellation between positive and negative error values, we take the square of each distance; we get a positive number each time that will add up to the total error. Such evaluation is very similar to what is done for the minimum chi-square estimation. The name "chi" comes from the Greek letter $\chi$, commonly used for the chi-square statistic. We will follow this protocol but in the 'machine learning way,' introducing a key concept: the cost function. 

## Cost Function in Linear Regression
The accuracy of the mapping function is measured by using a cost function. 
````{prf:definition}
:label: costFunction
The __cost function__ in linear regression returns a global error between the predicted values from a mapping function $h_\theta$ (predictions) and all the target values (observations) of the training data set.

It is also called __loss function__.

The commonly used cost function for linear regression, also called _squared error function_, or _mean squared error (MSE)_ is defined as:
```{math}
:label: costFunctionLinReg
 J\left(\theta_0, \theta_1\right) =\frac{1}{2 m} \sum_{i=1}^m\left(h_\theta\left(x_i\right)-y_i\right)^2
```
````
You can recognize the form of an average. The factor $\frac{1}{2}$ is to make it convenient when taking the derivative of this expression.
One can see from Equation {eq}`costFunctionLinReg` that each $h_\theta(x_i)$ is the prediction with our mapping function, whereas $y_i$ is the observed value in the data. 

The initial goal to "fit the data well" can now be formulated in a mathematical way: __find the parameters $\theta_0$ and $\theta_1$ that minimize the cost function__.
```{math}
:label: minCostFunction
\min_{\theta_0, \theta_1} J\left(\theta_0, \theta_1\right)
```

Let's simplify for now the problem by assuming the following data set:

```{glue:figure} plot_linReg_1234
:name: "plot_linReg_1234"
```
This is an ideal case for pedagogical purposes. What are the values of $\theta_0$ and $\theta_1$ here? 
```{admonition} Check your answers
:class: tip, dropdown
Recall the mapping function for linear regression: $h_\theta(x) = \theta_0 + \theta_1 x$. As we have a correspondance $y = 2x$ for all points, so $h_\theta(x) = 2x$, so $\theta_0 = 0$ and $\theta_1 = 2$.
```
But you will appreciate the simplification, as we will calculate the cost by hand for different values of $\theta_1$. More complicated things await you in the tutorial, promised.

```{admonition} In-class exercise
:class: seealso
* Start with a value of $\theta_1$ = 1 and calculate the cost function $J(\theta_1)$.
* Proceed the same for other values of $\theta_1$ of 0.5, 1.5, 2, 2.5, 3.
* How would the graph of the cost function $J(\theta_1)$ as a function of $\theta_1$ look like?
* Are there maxima/minima? If yes how many?
```

````{admonition} Solutions  |   Don't look too soon! Give it a try first.
:class: tip, dropdown  
The values of the cost function for each $\theta_1$ are reported on the plot below:
```{glue:figure} plot_linReg_costvstheta1
:name: "plot_linReg_costvstheta1"
```
We see that in this configuration, as we 'swipe' over the data points with 'candidate' straight lines, there will be a value for which we minimize our cost function. That is the value we look for.
````

Back to our original, more realistic exercise:

```{glue:figure} plot_linReg_50pts
:figwidth: 80%
:name: "plot_linReg_50pts"
```

How do we proceed to the minimization with two parameters? 

## Visualizing the cost
Let's have some visual display of our cost function as a function of our parameters $\theta$. We will make fancy 3D plots!



## Gradient Descent

Gradient descent is an iterative optimization algorithm to find the minimum of a function.










