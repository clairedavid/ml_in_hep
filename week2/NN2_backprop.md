# Backpropagation Algorithm

## Before Diving Into The Math 
### Ingredients

The forward propagation, or forward pass, will fill the network with values for all bias nodes and activation units. That includes the last layer of activation units, so the forward pass provide predictions. 

We saw the loss function as the mathematical tool to compare the predictions with their associated observed values for a sample (and the cost function aggregates this for all data samples).

Then we are familiar with the gradient descent procedure, which gives at each iteration the positive or negative amount to correct the weights to eventually get a model that fits to the data.

For a neural network, there are lots of knobs to tweak! Luckily, an efficient technique called backpropagation is able to compute the gradient of the network's error for every single model parameter.

### Definition

````{prf:definition}
:label: backpropdef
__Backpropagation__, short for __backward propagation of errors__, is an algorithm working from the output nodes to the input nodes of a neural network using the chain rule to compute how much each activation unit contributed to the overall error.

It automatically computes error gradients to then repeatedly adjust all weights and biases to reduce the overall error.
````

### Chain Rule Refresher

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



(NN2:backprop:mainstep)=
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
W^{(\ell)}_{q,r} &\leftarrow W^{(\ell)}_{q,r} - \alpha \frac{\partial \text{ Cost}}{\partial W^{(\ell)}_{q,r}} \\[1ex]
b^{(\ell)}_r &\leftarrow b^{(\ell)}_r - \alpha \frac{\partial \text{ Cost}}{\partial b^{(\ell)}_r}
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
Always a good question to start. We want to tweak the weights $\boldsymbol{W}$ and biases $\boldsymbol{b}$ so that the network predictions $\boldsymbol{\hat{y}}$ match the observed values $\boldsymbol{y}$. In other words, we want to know how the cost will change if the weights and biases change. For the neural network to fit the data, we need to find the weights and biases that minimize the cost function: 
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
Most mathematical textbooks have $x$ as the varying entity while explaining the derivative business. It can be here misleading with our notation as we use $\boldsymbol{x}$ as well. But in our case, $\boldsymbol{x}$ are the input features. They are given. They will not change (unless you bring new data, but there will still be given numbers you're not supposed to tweak). What we want is to vary the weights $\boldsymbol{W}$ and biases $\boldsymbol{b}$ to find optimized values for which the error is minimum, i.e. the model predicts a (given) $\hat{y}$ very close to the real target $y$.

```

Equation {eq}`partialdevcostWbeq` can be overwhelming, especially given the numerous quantities of weights and biases in a neural network. No panic! Thanks to backpropagation, there will be a way to not only get those derivatives, but also be very efficient in their computation.  


### Notations
Let's first rewrite the activation unit equation as a function of a function:
```{math}
:label: activvalueeq
\boldsymbol{a^{(\ell)}} = f\left( \; \boldsymbol{a^{(\ell -1)}} \;\boldsymbol{W^{(\ell)}} \;+\; \boldsymbol{b}^{(\ell)} \;\right) \;,
```
with $f$ the node's activation function and $\ell$ is the current layer of the activation unit where the sum is computed (taking thus as inputs the weights and biases pointing to that layer, see Figure {numref}`lec05_4_nn_notations`).
````{margin}
To further lighten the equations, parenthesis on the upperscripts for the layer number have been removed. It is fine here as the entities will not be squared but one should be careful to not mix the upperscript with an exponent.
````
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

The upperscripts for the weights and biases are not written inside the parenthesis for lighter equations. And it is also implied that $W$ and $b$ are collections of matrices and vectors respectively. 

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
***  
So the cost function is obtained using Equations {eq}`costfunceq` and {eq}`ypredwrappereq`:
```{math}
:label: costlossafzeq
\text{Cost} = \textstyle \sum L(f(z^L(W,b)))
```
***  
Let's joyfully take the derivatives of that sandwich of function! Now you get the chain rule refresher. The general expression for the partial derivative with respect to the weights $W$ would be:
```{math}
:label: dCostchaineq
\frac{\partial \text { Cost }}{\partial W}= \sum \; \frac{\partial L(f(z(W, b)))}{\partial f(z(W, b))} \cdot \; \frac{\partial f(z(W, b))}{\partial z(W, b)}  \cdot \; \frac{\partial z(W, b)}{\partial W}
```
A similar one can be written with $b$.  
Let's now go backward and see how it simplifies itself layer after layer.


### The backward walk
As its name indicates, the backward propagation proceed from the last to the first input layer. Let's write the general equation {eq}`dCostchaineq` for the last layer:
```{math}
:label: dCostlastchaineq
\begin{align*}
\frac{\partial \text { Cost }}{\partial W^L} &=\sum \; \frac{\partial L(f(z^L(W, b)))}{\partial f(z^L(W, b))}&\cdot& \; \frac{\partial f(z^L(W, b))}{\partial z^L(W, b)} &\cdot& \; \frac{\partial z^L(W, b))}{\partial W^L} 
\end{align*}
```

We can simplify things. The first term is the derivative of the loss function with $f(z^L(W, b)) = a^L$ as argument. It's a value here, computed with all weights values. Same for the second term: it is the derivative of activation function taken for the value $z^L$. For the third, we use the definition in Equation {eq}`zfunceq` that yields: $\left(z^L(W, b)\right)^{\prime} = a^{L-1}$. We can write:

```{math}
:label: dCostlastsimpleeq
\frac{\partial \text { Cost }}{\partial W^L}
= \; \sum \;\; L^{\prime}(a^L) 
\;\cdot \; f^{\prime}(z^L) 
\;\cdot \; a^{L-1}
```

Now let's proceed to the before last layer. Using the chain rule as usual:
```{math}
:label: dCostbeforelastchaineq
\begin{align*}
& \frac{\partial \text { Cost }}{\partial W^{L-1}} =  \\[1ex]
& \sum \; \frac{\partial L(f(z^L(W, b)))}{\partial f(z^L(W, b))} \cdot \frac{\partial f(z^L(W, b))}{\partial z^L(W, b)} \cdot \frac{\partial z^L(W, b))}{\partial a^{L-1}(W,b)} \cdot \frac{\partial a^{L-1}(W,b)}{\partial z^{L-1}(W,b)} \cdot \frac{\partial z^{L-1}(W,b)}{\partial W^{L-1}} 
\end{align*}
```

The two first terms are identical as in Equation {eq}`dCostchaineq`. Using the definitions of $a$ and $z$ we have: 
```{math}
:label: beforeLasttermssimplereq
\begin{gathered}
\frac{\partial z^L(W, b))}{\partial a^{L-1}(W,b)} \;=\; W^L  \qquad \frac{\partial a^{L-1}(W,b)}{\partial z^{L-1}(W,b)} \;=\; f'(z^{L-1}) \qquad  \frac{\partial z^{L-1}(W,b)}{\partial W^{L-1}} \;=\; a^{L-2}
\end{gathered}
```

Therefore:
```{math}
:label: dCostbeforelastsimpleeq
\frac{\partial \text { Cost }}{\partial W^{L-1}}
= \; \sum \;\; L^{\prime}(a^L) 
\;\cdot \; f^{\prime}(z^L) 
\;\cdot \; W^L 
\;\cdot \; f'(z^{L-1})
\;\cdot \; a^{L-2}
```

You can check yourself that for the derivative with respect to $W^{L-2}$ we will have: 
```{math}
:label: dCostbeforebeforelastsimpleeq
\frac{\partial \text { Cost }}{\partial W^{L-2}}
= \; \sum \;\; L^{\prime}(a^L) 
\;\cdot \; f^{\prime}(z^L) 
\;\cdot \; W^L 
\;\cdot \; f'(z^{L-1})
\;\cdot \; W^{L-1} 
\;\cdot \; f'(z^{L-1}) 
\;\cdot \; a^{L-3} 
```
We can see a pattern here! 

We go all the way to the first hidden layer 1 (scroll to the right):
```{math}
:label: dCostW1dotsxeq
\frac{\partial \text { Cost }}{\partial W^{1}}
= \; \sum \;\; L^{\prime}(a^L) 
\;\cdot \; f^{\prime}(z^L) 
\;\cdot \; W^L 
\;\cdot \; f'(z^{L-1})
\;\cdot \; W^{L-1} 
\;\cdot \; f'(z^{L-1}) 
\;\cdots\; W^2
\;\cdot \; f'(z^{1})
\;\cdot \; x
```
where $a^0 = x$ as defined in the previous lecture in Equation {eq}`xisazeroeq`.

### Recursive error equation
We can write Equations {eq}`dCostlastsimpleeq`, {eq}`dCostbeforelastsimpleeq` and {eq}`dCostbeforebeforelastsimpleeq` by introducing an error term $\delta$. For the last layer it is defined as the product of the first two partial derivatives times the activation unit's value at the previous layer. For the following (previous) layers it would be:
```{math}
:label: deltasandpartialcostseq
\begin{align*}
\delta^L &=\; L^{\prime}(a^L) \cdot f^{\prime}(z^L) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial W^L} &=& \quad \delta^L \;\cdot\; a^{L-1}\\[2ex]
\delta^{L-1} &= \; \delta^L     \cdot\;  W^L \cdot\; f'(z^{L-1}) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial W^{L-1}} &=& \quad \delta^{L-1} \;\cdot\; a^{L-2}\\[2ex]
\delta^{L-2} &= \; \delta^{L-1} \cdot\; W^{L-1} \;\cdot\; f'(z^{L-2}) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial W^{L-2}} &=& \quad \delta^{L-2} \;\cdot\; a^{L-3}\\
\end{align*} 
```
This is recursive because errors from the current layer are used to evaluate error signals in a previous layer. We can write the recursive formula for any partial derivative in layer $\ell$ as:

```{math}
:label: partialdevrecueq
\frac{\partial \text {Cost}}{\partial W^{\ell}} = \;\; \delta^{\ell} \;\cdot\; a^{\ell-1}
```

__What about the biases?__  
This is left as exercise for training. 
```{admonition} Exercise
:class: seealso
Express the partial derivatives of the cost with respect to the biases $b^{\ell}$.

Hint: start with the last layer $L$ as done previously with the weights.
```

````{admonition} Check your answer
:class: tip, dropdown

The formula is essentially the same as for the weights, at the difference that the partial derivative of $z^\ell$ with respect to $b^\ell$ is 1:
\begin{equation*}
\frac{\partial z^\ell}{\partial b^\ell} = 1
\end{equation*}

Thus:
```{math}
:label: biaseq
\begin{align*}
\delta^L &=\; L^{\prime}(a^L) \cdot f^{\prime}(z^L) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial b^L} &=& \quad \delta^L  \\[1ex]
\delta^{L-1} &= \; \delta^L     \cdot\;  W^L \cdot\; f'(z^{L-1}) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial b^{L-1}} &=& \quad \delta^{L-1}  \\[1ex]
\delta^{L-2} &= \; \delta^{L-1} \cdot\; W^{L-1} \;\cdot\; f'(z^{L-2}) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial b^{L-2}} &=& \quad \delta^{L-2} \\[1ex]
& \quad\cdots & & \cdots & \\[1ex]
\delta^{1} &= \; \delta^{2} \cdot\; W^{2} \;\cdot\; f'(z^{1}) 
& \Rightarrow & \quad \frac{\partial \text {Cost}}{\partial b^{1}} &=& \quad \delta^{1} \\[1ex]
\end{align*} 
```
````

### Weights and biases update
After backpropagating, each weight an bias in the network are ajusted in proportion to how much they contribute to overall error.

````{margin}
The equations are different as in the section {ref}`NN2:backprop:mainstep` as we keep here the 'lite' notations introduced above. The indices referring to the row and column of each weight/bias are implied for smoother reading. But remember that $W$ and $b$ are matrices and vectors respectively.
````
```{math}
:label: weightbiasupdate
\begin{align*}
W^\ell &\leftarrow W^\ell - \alpha \frac{\partial \text{ Cost}}{\partial W^\ell} \\[1ex]
b^\ell &\leftarrow b^\ell - \alpha \frac{\partial \text{ Cost}}{\partial b^\ell}
\end{align*}
```

### Memoization (and it's not a typo)
This is a computer science term. It refers to an optimization technique to make computations faster, in particular by reusing previous calculations. This translates into storing intermediary results so that they are called again if needed, not recomputed. Recursive functions by definition reuse the outcomes of the previous iteration at the current one, so memoization is at play.  

Let's illustrate this point by writing the derivative equations for a network with one output layer and three hidden layers:

```{math}
:label: lastfoursimpleeq
\begin{align*}
\frac{\partial \text { Cost }}{\partial W^4} &= \sum \; 
{\color{OliveGreen}L^{\prime}(a^4) \cdot f^{\prime}(z^4)} \cdot a^{3} \\[2ex]
\frac{\partial \text { Cost }}{\partial W^{3}} &= \sum \; 
{\color{OliveGreen}L^{\prime}(a^4) \cdot f^{\prime}(z^4)} \cdot {\color{Cyan}W^4 \cdot f'(z^{3})} \cdot a^{2}  \\[2ex]
\frac{\partial \text { Cost }}{\partial W^{2}} &= \sum \; 
{\color{OliveGreen}L^{\prime}(a^4) \cdot f^{\prime}(z^4)} \cdot {\color{Cyan}W^4 \cdot f'(z^{3})} \cdot {\color{DarkOrange}W^{3} \cdot f'(z^{2})} \cdot a^{1} \\[2ex]
\frac{\partial \text { Cost }}{\partial W^{1}} &= \sum \; 
{\color{OliveGreen}L^{\prime}(a^4) \cdot f^{\prime}(z^4)} \cdot {\color{Cyan}W^4 \cdot f'(z^{3})} \cdot {\color{DarkOrange}W^{3} \cdot f'(z^{2})}  \cdot W^2 \cdot f'(z^{1}) \cdot x\\[2ex]
\end{align*}
```
The reoccuring computations are highlighted in the same colour. Now you can get a sense of the genius behind neural network: although there are many computations, a lot of calculations are reused as we move backwards through the network. With proper memoization, the whole process can be very fast. 


## Summary on backpropagation
The backpropagation of error is a recursive algorithm reusing the computations from last until first layer to compute how much each activation unit and bias node contribute to the overall error. The idea behind backpropagation is to share the repeated computations wherever possible. 
Let's write again the step filling the key equations in:

````{prf:algorithm} Backpropagation
:label: backpropalgosummary

__Inputs__  
Training data set $X$ of $m$ samples with each $n$ input features, associated with their targets $y$

__Hyperparameters__
* Learning rate $\alpha$
* Number of epochs $N$

__Start__


__Step 0:__ Weight initialization

__Step 1:__ Forward propagation on subset or all sample instances:
\begin{equation}
\hat{y} = a^L(W, b) = f(z^L(W,b))
\end{equation}

__Step 2:__ Computation of loss function, cost function and overall error:
\begin{equation}
\text{Cost} = \textstyle \sum L(f(z^L(W,b))) \qquad \delta^L =\; L^{\prime}(a^L(W,b)) \cdot f^{\prime}(z^L(W, b)) 
\end{equation}

__Step 3:__ Computation of all activation unit errors: 
\begin{equation}
\delta^\ell = \; \delta^{\ell+1} \;\cdot\; W^{\ell+1} \;\cdot\; f'(z^\ell(W, b))
\end{equation}
... and derivatives:
\begin{equation}
\frac{\partial \text {Cost}}{\partial W^{\ell}} = \; \delta^{\ell} \;\cdot\; a^{\ell-1}  \qquad \qquad \qquad \frac{\partial \text {Cost}}{\partial b^{\ell}} = \; \delta^{\ell}
\end{equation}

__Step 4:__ Gradient Descent steps to update weights & biases:
```{math}
\begin{align*}
W^\ell &\leftarrow W^\ell - \alpha \frac{\partial \text{ Cost}}{\partial W^\ell} \\[1ex]
b^\ell &\leftarrow b^\ell - \alpha \frac{\partial \text{ Cost}}{\partial b^\ell}
\end{align*}

```

End of epoch, repeat step 1 - 4 until/unless:

__Exit conditions:__
* Number of epochs $N^\text{epoch}$ is reached
* If all derivatives are zero or below a small threshold 
````

This is the end of this math intensive section. Now you know the math behind a neural network and hopefully get a sense of why they are so powerful and popular.
There will be training on this with a tutorial where you will code yourself a small neural network from scratch.

In the next lecture, we will see a much more convenient way to build a neural network using a dedicated library. We will introduce further optimization techniques proper to deep learning.


```{admonition} Exercise
:class: seealso
Now that you know the backpropagation algorithm, a question regarding the neural network initialization: what if all weights are first set to the same value? (not zero, as we saw, but any other constant)
```

````{admonition} Check your answer
:class: tip, dropdown
If the weights and biases are initialized to the same constant values $w$ and $b$, all activation units in a given layer will get the same signal $a = \sum_{j} w x_j + b$. As such, all nodes for that layer will be identical. Thus the gradients will be updated the same way. Despite having many neurons per layer, the network will act as if it had only one neuron per layer. Therefore, it is likely to fail to reproduce complex patterns from the data; it won't be that smart. For a feedforward neural network to work, there should be an asymmetric configuration for it to use each activation unit uniquely. This is why weights and biases should be initalized with random value to break the symmetry.
````

&nbsp;&nbsp;


```{admonition} Learn more
:class: seealso
The paper that popularized backpropagation, back in 1989:  
[D. Rumelhart, G. Hinton and R.Williams, _Learning representations by back-propagating errors_](https://www.nature.com/articles/323533a0)

Some good refresher:  
[Derivative of the chain rule on math.libretexts.org](https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/03%3A_Derivatives/3.06%3A_The_Chain_Rule)

Backpropagation explanation with different notation and a source of inspiration (thanks) from [ml-cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html)

```