# What is the Sigmoid Function?

## General definition

A sigmoid function refers in mathematics to a category of functions having a characteristic "S-shape" curve. Among numerous examples, the one commonly used in machine learning is the logistic function.

````{margin}
To avoid confusion with the fact we already use $x$ for the input features, the input variable in the definition oppostive is written with another letter, $z$.
````

````{prf:definition}
:label: 
The __logistic function__ is defined for $z \in \mathbb{R}$ as 

\begin{equation*}
f(z) = \frac{1}{1 + e^{-z}}
\end{equation*}
````

It looks like this:

```{glue:figure} sigmoid_example
:name: "sigmoid_example"
```


## A better mapping

In our example with the energy of the electron, we could see from the data (easily because it is only in one dimension) that the bigger the energy of the electron, the more likely it is for the event to be classified as signal. If we overlay the S-curve on the data points, we start seeing interesting things.

```{image} ../images/lec03_2_scatter1D_sigmoid.png
:alt: scatter1D_sigmoid
:width: 80%
:align: center
```  
  \
  \
First of all, the curve is not overshooting below or above our discrete outcomes' range. Second: for data points either far left or far right, instead of creating a large error with a straight line as previously, the S-curve actually takes the values of our target-variables (asymptotically). Consequence: the error will very small, even negligible. We will not have an unwanted shift and mis-classification like before. 




````{prf:definition}
:label:

For logistic regression using the linear equation
\begin{equation*}
z = \sum_{j=1}^n \theta_j x^{(i)}_j = \theta^{\; T} x^{(i)} \\
\end{equation*}
between input features $x^{(i)}$ and the parameters $\theta$ to optimize,  

we define the __mapping function__ $h_\theta(x)$ as the logistic function of the linear equation: 
\begin{equation*}
h_\theta (x^{(i)}) = f(z) = f(\theta^{\; T} x^{(i)}) = \frac{1}{1 + e^{- \theta^{\; T} x^{(i)}}}
\end{equation*}
````

The mapping function satisfies
```{math}
:label: 
0 < h_\theta (x^{(i)}) < 1
```
Those limits are reached asymptotically reaching 0 and 1 when $z \rightarrow -\infty$ and $z \rightarrow +\infty$ respectively.

Intuitively, we see in our example that events with very low electron energy are most likely to be background whereas events with high electron energy are more likely to be signal. In the middle, there is a 50/50 chance to mis-classify an event.

__The output of the sigmoid can be interpreted as a probability__. 

With this new tool at hand, let's now see how it is incorporated in a custom-made cost function for logistic regression.









