# Varying the Learning Rate

## Motivations
We saw in previous lectures that the Gradient Descent algorithm updates the parameters, or weights, in the form:
```{math}
:label: paramupdate
W_j \quad \leftarrow \quad W_j -\alpha \frac{\partial}{\partial W_j} J\left(W\right) \;   \;   \;  \;   \;   \; \forall j \in [0..n]
```
````{margin}
Recall that the learning rate $\alpha$ is the hyperparameter defining the step size on the parameters at each update.
````
The learning rate $\alpha$ is kept constant through the whole process of Gradient Descent.

But we saw that the model's performance could be drastrically affected by the learning rate value; if too small the descent would take ages to converge, too big it could explode and not converge at all. How to properly choose this crucial hyperparameter?

In the florishing epoch (pun intended) of deep learning, new optimization techniques have emerged. The two most influencial families are __Learning Rate Schedulers__ and __Adaptative Learning Rates__. 

## Learning Rate Schedulers
````{prf:definition}
:label: learningratescheduledef
In a training procedure involving a learning rate, a __learning rate scheduler__ is a method consisting of modifying the learning rate at the beginning of each iteration. 

It uses a __schedule function__, which takes the current learning rate as input, along with the iteration and epoch indices, to output a new learning rate.

The updated learning rate is used in the optimizer.
````

As the number of iterations and epochs 

```{warning}
__What is the difference between epoch, batches and iteration?__

As we saw in Lecture 2 Section {ref}`warmup:linRegGD:gradientDescent`, an epoch is equivalent to the number of times the algorithm scans the entire data. It is a pass seeing the entire dataset. The batch size being the number of training samples, an iteration will be the number of batches needed to complete one epoch.
Example: if we have 1000 training samples splits in batches of size 250 each, then it will take 4 iterations to complete one epoch.
```

```{admonition} Question
:class: seealso
How should the learning rate vary? Should it increase? Decrease? Both?
```

Below are common learning schedules.

### Power Scheduling
The learning rate is modified according to:
````{margin}
The $t$ argument is the iteration integer, akin to a time (indicating duration of the training).
````
```{math}
:label: powerscheduleeq
\alpha(t) = \frac{\alpha_0}{\left(1 + \frac{t}{s}\right)^c} \; ,
```
where $\alpha_0$ is the initial learning rate. The steps $s$ and power $c$ (usually 1) are hyperparameters. The learning rate drops at each step; e.g. after $s$ steps it is divided by 2 (assuming $c$ = 1).  

### Time-Based Decay Scheduling
The time-based learning rate scheduler is often the standard (e.g. the default implementation in Keras library). It is controlled by a decay parameter $d = \frac{\alpha_0}{N}$, where $N$ is the number of epochs. It is similar (a particular case even) to the power scheduling:
```{math}
:label: timebaseddecayeq
\alpha(t) = \alpha_0 \times \frac{1}{1 + d \times t} 
```
As compared to a linear decrease, time-based decay causes learning rate to decrease faster upon training start, and much slower later (see graph below).

### Exponential Decay Scheduling
The exponential decay is defined as:
```{math}
\alpha(t) = \alpha_0 \times 0.1^{-t/s} \; ,
```
which will get the learning rate to decrease faster upon training start, and much slower later.

```{figure} ../images/lec08_2_lr_decays.png
---
  name: lec08_2_lr_decays
  width: 90%
---
 . Evolution of the learning rate vs 'time' as the number of epochs for three schedulers: linear decrease, time-based decay and exponential decay.  
 <sub>Image: neptune.ai</sub> 
```

### Step-based Decay Scheduling
Also called piecewise constant scheduling, this approach first uses of a constant learning rate for a given number of epochs and then the learning rate is reduced:
````{margin}
The floor function `floor(x)` or $\lfloor x \rfloor$ returns the greatest integer less than or equal to $x$.
````
```{math}
\alpha(t) = \alpha_0 \times e^{ \left \lfloor \frac{t}{s} \right \rfloor }
```

```{figure} ../images/lec08_2_lr_step.png
---
  name: lec08_2_lr_step
  width: 90%
---
 . Step-based learning rate decay.  
 <sub>Image: neptune.ai</sub> 
```

```{admonition} Learn More
:class: seealso
* Guide to Pytorch Learning Rate Scheduling on [Kaggle](https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook)
```

## Adaptative Learning Rate 

An algorithm combining Stochastic Gradient Descent with a learning rate schedule would have been considered close to the state-of-the-art before... the arrival of much faster methods called adaptative optimizers. Instead of a separate scheduler, the optimization of the learning rate is directly embedded in the optimizer. These optimizers approximate the gradient using model internal feedback. In other words, they incorporate the history in the weight update. Huge advantage: they are almost parameter-free.

````{prf:definition}
:label: 
_Adaptative Learning Rates__ refer
An __Adaptative Learning Rate__ refers to varying the learning rate using feedback from the model itself.
````
As the variation of the learning rate is done by the optimizer, adaptative learning rates and adaptative optimizers are equivalent terms.

Below are brief descriptions of the most popular adaptative optimizers.

### Momentum Optimization
In physics, the momentum $\boldsymbol{p}$ is a vector obtained by taking the product of the mass and velocity of an object. It quantifies motion. In computing science, momentum refers to the direction and speed at which the parameters move - via iterative updates - through the parameter space. 





```{admonition} Learn More
:class: seealso
Let's go to the source!  
The paper introducing Adam in 2015: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
```






================

simulated annealing 


https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
