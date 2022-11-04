# Backpropagation Algorithm

## Summary of Ingredients and Definition

The forward propagation, or forward pass, will fill the network with values for all bias nodes and activation units. That includes the last layer of activation units, so the forward pass provide predictions. 

We saw the loss function as the mathematical tool to compare the predictions with their associated observed values for a sample (and the cost function aggregates this for all data samples).

Then we are familiar with the gradient descent procedure, which gives at each iteration the positive or negative amount to correct the weights to eventually get a model that fits to the data.

For a neural network, there are lots of knobs to tweak! Luckily, an efficient technique called backpropagation is able to compute the gradient of the network's error for every single model parameter.

````{prf:definition}
:label: backpropdef
__Backpropagation__, short for __backward propagation of errors__, is an algorithm working from the output nodes to the input nodes of a neural network using the chain rule to compute how much each activation unit contributed to the overall error.

It automatically computes error gradients to then repeatedly adjust all weights and biases to reduce the overall error.
````

A little interlude and refresher of the chain rule will not hurt.

````{prf:definition}
:label: chainruledef
Let $f$ and $g$ be functions. For all $𝑥$ in the domain of $g$ for which $g$ is differentiable at $x$ and $f$ is differentiable at $g(x)$, the derivative of the composite function:
\begin{equation}
h(x) = f(g(x)) 
\end{equation}
is given by
\begin{equation}
h'(x) = \frac{\mathrm{d} \; f(g(x)) }{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}} \;\cdot \;  \frac{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}}{\mathrm{d} \; x} = f'\left(g(x)\right) \;\cdot \; g'(x)
\end{equation}
````
You can see above from the colouring that the inserted denominator and numerator of the composed function cancel out. All good. Now your turn with three functions (and that will be useful for the rest of the lecture).

```{warning}
The prime notation $f'(\square)$ can be error-prone. It is the derivative with respect to $\square$ as the variable, i.e. as a block on its own (even if it depends on other variables).
\begin{equation*}
\frac{\mathrm{d} f(\square)}{\mathrm{d} \square} = f'(\square ) 
\end{equation*}
```

```{admonition} Exercise
:class: seealso
What would be the chain rule for three functions?
\begin{equation}
k(x) = h(f(g(x))) 
\end{equation}
```

````{admonition} Check your answer
:class: tip, dropdown
```{math}
:label: chainrule3funceq
\begin{align*}
k'(x) &= \; \frac{\mathrm{d} \; h(f(g(x)))}{\mathrm{{\color{BurntOrange}d}} \; {\color{BurntOrange}f(g(x))}} &\;\cdot\;& \frac{\mathrm{{\color{Peach}d}} \; {\color{Peach}f(g(x))} }{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}} &\;\cdot \;&  \frac{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}}{\mathrm{d} \; x} \\[1ex]
&= \; h'(f(g(x))) &\;\cdot\;& f'\left(g(x)\right) &\;\cdot \;& g'(x)\\
\end{align*}
```

Three functions: three derivative terms.  
We work from the outside first, taking one derivative at a time.
````




## Main Steps
Before diving into a more mathematical writing, let's just list the main steps of backpropagation. We will detail steps 2, 3 and 4 very soon:

````{prf:algorithm} Backpropagation
:label: backpropalgo

__Inputs__  
Training data set $X$ of $m$ samples with each $n$ input features, associated with their targets $y$

__Hyperparameters__
* Learning rate $\alpha$
* Number of epochs $N$

__Start__


__Step 0:__ Weight initialization

__Step 1:__ Forward propagation on subset or all sample instances $\Rightarrow$ get predictions $\boldsymbol{\hat{y}}$

__Step 2:__ Computation of loss function & cost function $\Rightarrow$ get network's output error $\delta^{(L)}$

__Step 3:__ Computation of all activation unit errors  $\Rightarrow$ $\delta^{(L-1)}_k, \delta^{(L-2)}_p, \cdots, \delta^{(1)}_r$ 

__Step 4:__ Gradient Descent steps to update weights & biases:
```{math}
\begin{align*}
W^\text{new} &= W^\text{old} - \alpha \frac{\partial \text{ Cost}}{\partial W} \\[1ex]
b^\text{new} &= b^\text{old} - \alpha \frac{\partial \text{ Cost}}{\partial b} 
\end{align*}

```

End of epoch, repeat step 1 - 4 until/unless:

__Exit conditions:__
* Number of epochs $N^\text{epoch}$ is reached
* If all derivatives are zero or below a small threshold 
````

## Computations

Now there will be math.

### What is the goal?
Always a good question to start. We want to tweak the weights $\boldsymbol{W}$ and biases $\boldsymbol{b}$ so that the network predictions $\boldsymbol{\hat{y}}$ match the observed values $\boldsymbol{y}$. 

For this we use the mathematical tool of the cost function. As there will be many letters in the following, let's write the cost in full for clarity. We need to find the weights and biases that minimize the cost function: 
```{math}
:label: costnnmineq
 \min_{\boldsymbol{W},\boldsymbol{b}} \text{ Cost}(\boldsymbol{W},\boldsymbol{b})
```

For this, we will need the partial derivatives of the cost function with respect to the parameters we want to optimize.

With derivatives, especially partial ones, it's crucial to ask the question:

What varies here and with respect to what?  
How will the numerator entity change as the denominator change?  
Here we want to know how to vary the weights and biases so that the cost gets lower:
```{math}
:label: partialdevcostWbeq
\begin{gathered}
\frac{\partial \text{ Cost}( \boldsymbol{W},\boldsymbol{b} )}{\partial \boldsymbol{W}} \qquad \frac{\partial \text{ Cost}( \boldsymbol{W},\boldsymbol{b} )}{\partial \boldsymbol{b}}
\end{gathered}
```

```{warning}
Most mathematical textbooks have $x$ as the varying entity while explaining the derivative business. It can be here misleading with our notation as we use $\boldsymbol{x}$ as well. But in our case, $\boldsymbol{x}$ are the input features. They are given. They will not change (unless you bring new data, but there will still be given numbers you're not supposed to tweak). What we want is to vary the weights $\boldsymbol{W}$ and biases $\boldsymbol{b}$ to find optimized values for which the error is minimum, i.e. the model predicts a (given) $\hat{y}$ very close from the real target $y$.

```

Equation {eq}`partialdevcostWbeq` can be overwhelming, especially given the numerous quantities of weights and biases in a neural network. No panic! Thanks to backpropagation, there will be a way to not only get those derivatives, but also be very efficient in their computation.  


### Notations
Let's first rewrite the activation unit equation as a function of a function:
```{math}
:label: activvalueeq
\boldsymbol{a^{(\ell)}} = f\left( \; \boldsymbol{a^{(\ell -1)}} \;W^{(\ell)} \;+\; \boldsymbol{b}^{(\ell)} \;\right) \;,
```
with $f$ the node's activation function and $\ell$ is the current layer of the activation unit where the sum is computed (taking thus as inputs the weights and biases pointing to that layer, see Figure {numref}`lec05_4_nn_notations`).

Let $z$ be a function for the "weighted sum plus bias:"
```{math}
:label: zfunceq
z^\ell(W, b) = a^{\ell-1} W^\ell + b^\ell
```

Now we can rewrite the output from an activation unit $a^{\ell}$ as:
```{math}
:label: afunczeq
a^\ell(W, b) = f\left( z^\ell(W, b) \right)
```
````{margin}
To further lighten the equations, parenthesis on the upperscripts for the layer number have been removed. It is fine here as the entities will not be squared but one should be careful to not mix the upperscript with an exponent.
````
The upperscripts for the weights and biases are not written inside the parenthesis for lighter equations, but it is implied that the function is applying to the $W^{(\ell)}$ and $b^{(\ell)}$. And it is also implied that W and b are collections of matrices and vectors respectively. 

We will denote the loss function through a general form as $L$:
```{math}
:label: lossfunceq
L(\hat{y}^{(i)}, y^{(i)}) 
```
It is computed for each sample instance $\left\{ \boldsymbol{x^{(i)}}, y^{(i)} \right\}$, with $\boldsymbol{x^{(i)}}$ being one row of input features and $y$ the associated target. 

The cost is the sum of the losses over all data instances $m$. To lighten the equations of the following section, the sum will be written without instance indices as upperscript (already used for the layer number); it is implied that it is the sum over all data instances.

```{math}
:label: costfunceq
\text{Cost} = \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)}) = \textstyle \sum L(\boldsymbol{\hat{y}}, \boldsymbol{y})
```

__How can we express the final output $\boldsymbol{\hat{y}}$?__
Let's take a similar network as the one in the previous lecture but layers are labeled from the last one (right) in decreasing order:
```{figure} ../images/lec07_3_nn_lastlayers.png
---
  name: lec07_3_nn_lastlayers
  width: 100%
---
 . Feedforward neural network with the notations for the  
 last, before last and before before last layers.    
<sub>Image from the author</sub>
```

The final prediction $\hat{y}$ is the output of the activation unit in the last layer:
```{math}
:label: ypredwrappereq
\hat{y} = a^L = f(z^L(W,b))
```

So the cost function is obtained using Equations {eq}`costfunceq` and {eq}`ypredwrappereq`:
```{math}
:label: costlossafzeq
\text{Cost} = \textstyle \sum L(f(z^L(W,b)))
```

Let's joyfully take the derivatives of that sandwich of function! Now you get the chain rule refresher. The expression for the partial derivative with respect to the weights $W$ would be:

```{math}
:label: dCostchaineq
\begin{align*}
\frac{\partial \text { Cost }}{\partial W}&=
\sum \; \frac{\partial L(f(z(W, b)))}{\partial f(z(W, b))}
&\cdot& \; \frac{\partial f(z(W, b))}{\partial z(W, b)} 
&\cdot& \; \frac{\partial z(W, b))}{\partial W}
\end{align*}
```

Let's now go backward and see how it simplifies itself layer after layer.


### The backward walk
As its name indicates, the backward propagation proceed from the last to the first input layer. Let's write the general equation {eq}`dCostchaineq` for the last layer:
```{math}
:label: dCostlastchaineq
\begin{align*}
\frac{\partial \text { Cost }}{\partial W^L} &=\sum \; \frac{\partial L(f(z^L(W, b)))}{\partial f(z^L(W, b))}&\cdot& \; \frac{\partial f(z^L(W, b))}{\partial z^L(W, b)} &\cdot& \; \frac{\partial z^L(W, b))}{\partial W^L} 
\end{align*}
```

We can simplify things. The first derivative of the loss function takes $f(z^L(W, b)) = a^L$ as argument. It's a value here, computed with all weights values. Same for the second derivative: it is $z^L$, a number. For the third, we use the definition of Equation {eq}`zfunceq` that yield the value $a^{L-1}$. We can write:

```{math}
:label: dCostlastsimpleeq
\frac{\partial \text { Cost }}{\partial W^L}
= \; \sum \;\; L^{\prime}(a^L) 
\;\cdot \; f^{\prime}(z^L) 
\;\cdot \; a^{L-1}
```

Now let's proceed to the before last layer. Using the chain rule as usual:









```{admonition} Exercise
:class: seealso
Now that you know the backpropagation algorithm, a question regarding the neural network initialization: what if all weights are first set to the same value? (not zero, as we saw, but any other constant)
```

````{admonition} Check your answer
:class: tip, dropdown
If the weights and biases are initialized to the same constant values $w$ and $b$, all activation units in a given layer will get the same signal $a = \sum_{j} w x_j + b$. As such, all nodes for that layer will be identical. Thus the gradients will be updated the same way. Despite having many neurons per layer, the network will act as if it had only one neuron per layer. Therefore, it is likely to fail to reproduce complex patterns from the data; it won't be that smart. For a feedforward neural network to work, there should be an asymmetric configuration for it to use each activation unit uniquely. This is why weights and biases should be initalized with random value to break the symmetry.
````


```{admonition} Learn more
:class: seealso
The paper that popularized backpropagation, back in 1989:  
[D. Rumelhart, G. Hinton and R.Williams, _Learning representations by back-propagating errors_](https://www.nature.com/articles/323533a0)

Some good refresher:  
[Derivative of the chain rule on math.libretexts.org](https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/03%3A_Derivatives/3.06%3A_The_Chain_Rule)
```



===================
The idea behind backpropagation is to share the repeated computations wherever possible. We will see this in details soon. Let's first list the main steps.

[application of Chain rule to find the Derivatives of cost with respect to any variable in the nested equation]


Thanks ml-cheatsheet