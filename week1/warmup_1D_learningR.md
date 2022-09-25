(warmup:lr)=
# Learning Rate
The learning rate decides on the degree of parameter update.
The learning rate $\alpha$ is a hyperparameter intervening in the calculation of the step size at which the parameters will be incremented or decremented.

## Learning Rate and Convergence
The learning rate is not directly setting the step size. It is a coefficient. With a fixed $\alpha$, the gradient descent can converge as the steps will become smaller and smaller due to the fact that the derivatives $\frac{\partial }{\partial \theta_j} J(\theta)$ will get smaller (in absolute value) as much as we approach the minimum:

````{margin}
Recall that the step size is given by $-\alpha \frac{d}{d \theta} J(\theta) = -\alpha \times$ _slope_. Here the magnitude of the slope is decreasing at each iteration. Eventually, $\frac{d}{d \theta} J(\theta) \rightarrow 0$ so $\theta' = \theta - \alpha \times 0 \rightarrow \theta $. 
````
```{figure} ../images/lec02_3_smaller_steps.png
---
  name: lec02_3_smaller_steps
  width: 80%
---
. Here the step size is reduced at the next iteration of the gradient descent, even if $\alpha$ remains constant. <sub>Image from the author</sub>
 ```

## Learning Rate and Divergence
A learning rate too big will generate an updated parameter on the other side of the slope.  
Two cases:
* __The zig-zag__: if the next parameter $\theta'$ is at a smaller distance to the $\theta^{\min J}$ minimizing the cost function ($ | \theta' - \theta^{\min J} |  < | \theta - \theta^{\min J} |$), the gradient descent will generate parameters oscillating on each side of the slope until convergence. It will converge, but it will require a lot more steps. 
* __Divergence__: if the next paremeter is at a greater distance than the $\theta^{\min J}$ minimizing the cost function ($ | \theta' - \theta^{\min J} |  > | \theta - \theta^{\min J} |$), the gradient descent will produce new parameters further and further away, escaping the parabola! It will diverge. We want to avoid this.

The divergence is illustrated on the right in the figure below: 
```{figure} ../images/lec02_3_learningRate_small_big.jpg
---
  name: lec02_3_learningRate_small_big
  width: 100%
---
. The learning rate determines the step at which the parameters will be updated (left). Small enough: the gradient descent will converge (middle). If too large, the overshoot can lead to a diverging gradient, no more "descending" towards the minimum (right). <sub>Image from [kjronline.org](https://www.kjronline.org/ViewImage.php?Type=F&aid=658625&id=F7&afn=68_KJR_21_1_33&fn=kjr-21-33-g007_0068KJR)</sub>
 ```


## Summary
* The learning rate $\alpha$ is a hyperparameter intervening in the calculation of the step size at which the parameters will be incremented or decremented.
* The step size varies even with a constant $\alpha$ as it is multiplied by the slope, i.e. the derivatives of the cost function.
* A small learning rate is safe as it likely leads to convergence, yet too small values will necessitates a high number of epochs.
* A large learning rate can overshoot the minimum of the cost function and lead to either 
  * an oscillating trajectory of the parameters: it converges yet with more iterations are needed
  * a diverging path: the gradient descent fails to converge toward the minimum value of the cost function


```{admonition} Question
:class: seealso
How to choose the best value for the learning rate?
```
We will discuss your guesses. Then the next section will give you the tricks!