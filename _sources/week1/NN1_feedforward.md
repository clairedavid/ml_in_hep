# Feedforward Propagation


## What is Feedforward Propagation?

It is a first step in the training of a neural network (after initialization of the weights, which will be covered in the next lecture). The forward direction means going from input to output nodes. 

````{prf:definition}
:label: feedforwardpropdef
The __Feedforward Propagation__, also called __Forward Pass__, is the process consisting of computing and storing all network nodes' output values, starting with the first hidden layer until the last output layer, using at start either a subset or the entire dataset samples.
````

Forward propagation thus leads to a list of the neural network prediction for each data instance used in the input. At each node, the computation is the key equation {eq}`aneq` we saw in the previous Section {ref}`NN1:modelRep`, written again for convenience:
```{math}
:label: sumwixieq
y = f\left(\sum_{j=1}^n w_j x_j + b \right)
```

But there will be some change in the notations. Let's define everything in the next subsection.

## Notations
Let's write the following network with $x_n$ input features, one first hidden layer with $a_q$ activation units and a second one with $a_r$ activation units. For simplicity, we will choose an output layer with only one node:


```{figure} ../images/lec05_4_nn_notations.png
---
  name: lec05_4_nn_notations
  width: 100%
---
 . A feedforward neural network with the notation we will use for the forward propagation equations (more in text).    
<sub>Image from the author</sub>
```

There are lots of subscripts and upperscripts here. Let's explain the conventions we will use.  

__Input data__  
We saw in Lecture 2 that the dataset in supervised learning can be represented as a matrix $X$ of $m$ data instances (rows) of $n$ input features (columns). For clarity in the notations, we will focus for now on only one data instance, the $i$th sample row $\boldsymbol{x^{(i)}} = ( x_1, x_2, \cdots, x_n)$.

__Activation units__
In a given layer $\ell = 1, 2, \cdots, N^\text{layer}$, the activation units will give outputs that we will note as a row vector 
```{math}
\boldsymbol{a^{(\ell)}} = ( a_1^{(\ell)}, a_2^{(\ell)}, \cdots , a_q^{(\ell)}), 
```
where the upperscript is the layer number and the subscript is the row of the activation unit in the layer, starting from the top.
````{margin}
The layer numbering starts at the first hidden layer where $\ell=1$. The input layer is $\ell=0$.
````
__Biases__  
The biases are also row vectors, one for each layer it connects to.  
```{math}
\boldsymbol{b^{(\ell)}} = ( b_1^{(\ell)}, b_2^{(\ell)}, \cdots , b_q^{(\ell)})
```
If the last layer is only made of one node like in our example above, then $b$ is a scalar. 

__Weights__  
Now the weights. You may see in the literature different ways to represent them. In here we use a convention we could write as:
```{math}
w^\ell_{(\ell -1) \; \to \; \ell}
```
For instance $w^{(2)}_{3,1}$ is the weight from the third node of layer (1) going to the first node of layer (2). In other words, the first index is the row of the node from the previous layer (departing node of the weight's arrow) and the second index is the row of the node from the current layer (the one the weight's arrow points to).  

We can actually represent each weight from layer $\ell -1$ to layer $\ell$ as a matrix $W^{(\ell)}$. If the previous layer $\ell -1$ as $n$ nodes and the layer $\ell$ has $q$ activation units, we will have:

```{math}
:label: Wmatrixeq
W^{(\ell)} = \begin{pmatrix}
w_{1,1}^{(\ell)} & w_{1,2}^{(\ell)} & \cdots & w_{1,q}^{(\ell)} \\[2ex]
w_{2,1}^{(\ell)} & w_{2,2}^{(\ell)} & \cdots & w_{2,q}^{(\ell)} \\[1ex]
\vdots  & \vdots & \ddots   & \vdots \\[1ex]
w_{n,1}^{(\ell)} & w_{n,2}^{(\ell)} &  \cdots & w_{n,q}^{(\ell)} \\
\end{pmatrix}
```

Let's now see how we calculate all the values of the activation units!


## Step by step calculations

### Computation of the first hidder layer 
Let's use Equation {eq}`sumwixieq` to compute the activation unit outputs of the first layer. The activation function is represented as $f$ here:
```{math}
:label: firstlayereq
\begin{align*}
a^{(1)}_1 &= f\left(\; w_{1,1}^{(1)} \; x_1 \;+\; w_{2,1}^{(1)} \; x_2 \;+\; \cdots + \; w_{n,1}^{(1)} \; x_n \;+\; b^{(1)}_1\right)\\[2ex]
a^{(1)}_2 &= f\left(\; w_{1,2}^{(1)} \; x_1 \;+\; w_{2,2}^{(1)} \; x_2 \;+\; \cdots + \; w_{n,2}^{(1)} \; x_n \;+\; b^{(1)}_2\right)\\
&\vdots \\[2ex]
a^{(1)}_q &= f\left(\; w_{1,q}^{(1)} \; x_1 \;+\; w_{2,q}^{(1)} \; x_2 \;+\; \cdots + \; w_{n,q}^{(1)} \; x_n \;+\; b^{(1)}_q\right)\\
\end{align*}
```
We can actually write it in the matrix form. The expanded version being:
```{math}
:label: firstlayermatrixexpandedeq
\boldsymbol{a^{(1)}} = f\left[ \; ( x_1, x_2, \cdots, x_n) 
\begin{pmatrix}w_{1,1}^{(1)} & w_{1,2}^{(1)} & \cdots & w_{1,q}^{(1)} \\[2ex]w_{2,1}^{(1)} & w_{2,2}^{(1)} & \cdots & w_{2,q}^{(1)} \\[1ex]\vdots  & \vdots & \ddots   & \vdots \\[1ex]w_{n,1}^{(1)} & w_{n,2}^{(1)} &  \cdots & w_{n,q}^{(1)} \\\end{pmatrix}
 \;+\; ( b_1^{(1)}, b_2^{(1)}, \cdots , b_q^{(1)}) \; \right]
```

And the compact one:
```{math}
:label: firstlayermatrixeq
\boldsymbol{a^{(1)}} = f\left( \; \boldsymbol{x} \;W^{(1)} \;+\; \boldsymbol{b}^{(1)} \;\right)
```
Much lighter. 

### Computation of the second hidder layer 
Let's do the same calculation for the second layer of activation units. Instead of the dataset vector $\boldsymbol{x}$, we will have $\boldsymbol{a^{(1)}}$ as input:
```{math}
:label: secondlayermatrixexpandedeq
\boldsymbol{a^{(2)}} = f\left[ \; ( a^{(1)}_1, a^{(1)}_2, \cdots, a^{(1)}_q) 
\begin{pmatrix}w_{1,1}^{(2)} & w_{1,2}^{(2)} & \cdots & w_{1,r}^{(2)} \\[2ex]
w_{2,1}^{(2)} & w_{2,2}^{(2)} & \cdots & w_{2,r}^{(2)} \\[1ex]
\vdots  & \vdots & \ddots   & \vdots \\[1ex]
w_{q,1}^{(1)} & w_{q,2}^{(1)} &  \cdots & w_{q,r}^{(1)} \\
\end{pmatrix} \;+\; ( b_1^{(2)}, b_2^{(2)}, \cdots , b_r^{(2)}) \; \right]
```

And the elegant, light version:
```{math}
:label: firstlayermatrixeq
\boldsymbol{a^{(2)}} = f\left( \; \boldsymbol{a^{(2)}} \;W^{(2)} \;+\; \boldsymbol{b}^{(2)} \;\right)
```

We start seeing a pattern here thanks to the matricial form. 

### Computation of the third hidder layer 


## General... 

=================


Nice animation! https://yogayu.github.io/DeepLearningCourse/03/ForwardPropagation.html