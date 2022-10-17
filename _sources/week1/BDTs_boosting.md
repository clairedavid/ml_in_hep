# What is boosting?

## Definitions

The general idea behind boosting is a correction of previous learners by the next row of classifiers.

````{prf:definition}
:label: boostingdef
__Boosting__ refers to ensemble methods that are tuning weak learners into a strong one, usually sequentially, with the next predictor correcting its predecessors.
````
Usually the predictors are really shallow trees, namely one root nodes and two final leaves. This has a name:
````{prf:definition}
:label: decisionstumpdef
A __decision stump__ is a one-level decision tree. It has one root node evaluating only one input feature and the two resulting branches immediately connect to two terminal nodes, i.e. leaves.
````

Many boosting methods are available. We will see two popular ones: AdaBoost and Gradient Boosting.

## AdaBoost

### Algorithm
AdaBoost is short for Adaptative Boosting. It works by assigning larger weights to data samples misclassified by the previous learner. Let's see how this work.

````{prf:algorithm} AdaBoost
:label: AdaBoostalgo


__Inputs__  
Training data set $X$ of $m$ samples

__Outputs__  
A collection on decision boundaries segmenting the $k$ feature phase space.

__Initialization__  
Each training instance $x^{(i)}$ is given the same weight 
\begin{equation*}
w^{(i)} = \frac{1}{m}
\end{equation*}

__Start__  
__For__ each predictor $j = 1 , \cdots , N^\text{pred}$  
  a. Train on all samples and compute the weighted error rate $r_j$ 
   \begin{equation}
   r_j = \frac{\sum_{i = 1}^m w^{(i)} [ \hat{y}_j^{(i)} \neq y^{(i)} ]   }{\sum_{i = 1}^m w^{(i)}}
   \end{equation}
  b. Give the predictor $j$ a weight $W_j$ measuring accuracy  
   \begin{equation}
   W_j = \alpha \log \frac{1 - r_j}{r_j} 
   \end{equation}
   $W_j$ points to zero if the predictor is bad, or a high number if the predictor is good.  
   $\alpha$ is the learning rate.
   
  c. Update the weights of all data samples:
   \begin{equation}
   w^{(i)} = \left\{\begin{matrix}
   w^{(i)} \;\;\;\;  &\text{if} \;\;\;\; \hat{y}_j^{(i)} = y_j^{(i)} \\[2ex]
   \;\;w^{(i)} \exp (W_j)  \;\;\;\;  &\text{if} \;\;\;\; \hat{y}_j^{(i)} \neq y_j^{(i)} \\
   \end{matrix}\right.
   \end{equation}
  d. Normalize the weights
   \begin{equation}
    w^{(i)} \rightarrow \frac{w^{(i)}}{\sum_{i = 1}^m w^{(i)}}
   \end{equation}

__Exit conditions__  
* $N^\text{pred}$ is reached
* All data sample are correctly classified (perfect classifier)

````

The illustration below gives a visual of the algorithm.

```{figure} ../images/lec04_3_adaboost.png
---
  name: lec04_3_adaboost
  width: 80%
---
 . Visual of AdaBoost.  
 Misclassified samples are given a higher weight for the next predictor.  
 Base classifiers are decision stumps (one-level tree).  
 <sub>Source: subscription.packtpub.com</sub>
 ```

```{note}
As the next predictor needs input from the previous one, the boosting is not an algorithm that can be parallelized on several cores but demands to be run in series.
```


How is the algorithm making predictions? In other words, how are all the decision boundaries (cuts) combined into the final boosted learner?  

The combined prediction is the class obtaining a weighted majority-vote, where votes are weighted with the predictor weights $W_j$.

```{math}
:label: 
\hat{y}(x^\text{new}) = \arg_k \max \; \; \sum_{j = 1}^{N^\text{pred}} W_j \;[\; \hat{y}_j(x^\text{new}) = k \;]
```

### Implementation

In Scikit-Learn, the AdaBoost classifier can be implemented this way:

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier( 
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5)

ada_clf.fit(X_train, y_train)
```

The decision trees are very 'shallow' learners: only a root note and two final leaf nodes (that's what a max depth of 1 translates to). But there are usually a couple of hundreds of them. The `SAMME` acronym stands for Stagewise Additive Modeling using a Multiclass Exponential Loss Function. It's nothing else than an extension of the algorithm where there are more than two classes. The `.R` stands for Real and it allows for probabilities to be estimated (predictors need the option `predict_proba` activated, otherwise it will not work).


```{admonition} Learn More
:class: seealso
* [AdaBoost on Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
```

## Gradient Boosting
The gradient is back! 

... 
