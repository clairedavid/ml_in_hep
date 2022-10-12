# Bias, Variance: how to cope

How to evaluate the performance of a machine learning algorithm?  
How to detect the presence of underfitting or overfitting?  
How to tune the algorithm to get a good fit?
 
## Cross-validation
In supervised machine learning, we have access to data containing labels, i.e. the input features have associated targets. Whether it is regression or classification, we know the answer. To assess how the model will deal with new cases, we need to compare its predictions with the answers. We cannot perform this comparison if we don't have the labels! To cope, the input data set is split into different data subsets, each corresponding of a step in the optimization workflow:

````{prf:definition}
:label: trainvaltestsetsdef

The __training set__ is a subset of the input data dedicated to the fitting procedure to find the model parameters minimizing the cost function (step 1).

The __validation set__ is used to assess the performance of the model and tune the model's hyperparameters (step 2).

_Multiple models with various hyperparameters are iteratively trained again using __only__ the training set, then validated using __only__ the validation test until a given satisfying performance is achieved. The model of higher performance goes to the test step._

The __test set__ is the final assessment done on the model; the resulting error rate on the test set is called __generalization error__, an estimate on the errors for future predictions with new data samples (step 3).
````
The general split between the training, validation and test subsets are 60%, 20% and 20%. But depending on the number of samples, a smaller test set is sufficient. Reducing it allows for an increase of the training and validation set sizes, exposing the model to more data samples for training and validation.

```{warning}
The terms _validation_ and _test_ are sometimes interchangeably used both in industry and in academia, creating some terminological confusion. What is important to keep in mind is that once the iterative back and forth of steps 1 and 2 are giving a most performing model, the hyperparameters of the model are frozen and the last check, step 3, is done on not-yet-seen data with those hyperparameters.  

TL;DR: Never use the final test data for tuning.
```

Data sets are split randomly, by shuffling the data rows and cutting and taking the table indices corresponding to the relative split between the three sub-collections. But precious information is lost by not training on the entire data samples available. Moreoever, one of the subset could pick more random outliers or noisy features that will deteriorate either the training or validation outcomes. To cope with this, a commonly used technique is to use the training set as validation set and vice versa, then pick the best performing outcome (set of hyperparameters). For instance if the entire data sample is split in terms of train/validate/test as A/B/C/D, with D the final test set, the cross validation would consist of:
* training with sets (BC) and validate with set A
* training with sets (CA) and validate with set B
* training with sets (AB) and validate with set C

In this example, we have a train/validate split of three sub-sets, we talk of a 3-fold cross-validation. The general name with $k$ sub-sets is $k$-fold cross validation.

````{prf:definition}
:label: kfoldxvalidation
The $\mathbf{k}$__-fold cross-validation__ is a procedure consisting of using $k$ subsets of the data used for the training and validation steps where each subset is rotationally used as the validation set, while the $k-1$ other subsets are merged as one training set.  

The $k$ validation results are combined to provide an estimate of the model's predictive performance.
````
With $k$-fold cross-validation, the estimate of the model's predictive performance comes with a precision (the standard deviation from the collection of $k$ estimates). However cross-validation necessitates more computing time yet it is more robust against noise or outliers picked by the random splitting. 

We now know how to manipulate our data set to get an estimate of performance. But what is this quantifier exactly? How to visualize it?

## Types of errors
Generally speaking, an error is the gap between the prediction and the true value. There are different terms depending on the algorithm and the set on which errors are computed. 

### For regression algorithms
````{prf:definition}
:label: residualDef
A __residual__ is the difference between the actual and predicted value computed with respect to data samples used to train, validate and tune the model (i.e. _in-sample_ error).
````
It is always a good practice to make a scatter plot of all the residuals with respect to the independent variable values; they should be randomly distributed on a band symmetrically centered at zero. If not, this means the chosen model is not appropriate to correctly fit the data.

````{prf:definition}
:label: errorDef
The __generalized error__ is the error rate of the difference between the actual and predicted values computed with respect to test or new data (i.e. _out-of-sample_ error).
````

### For classification algorithms
In classification, the errors bear different names. As we saw that multiclassifiers are treated as a collection of binary classifiers, we will go over the two types of errors. 
The most common metric is the root mean squared error (RMSE).

Recall than the labelling and numerical association (1 and 0) of classes is arbitrary. Signal can be the rare process we want to see in the detector and 0 the background we want to reject. But we could exchange the numbers with 0 and 1, provided we remain consistent. In medical diagnosis, the class labelled 1 can be the presence of cancer on a patient (it's not what we want of course, but what we are looking to classify). 

````{prf:definition}
:label: confusionMatrix def
The __confusion matrix__ is a table used to visualize the prediction results (positive/negative) from a classification algorithm with respect to their correctness (true/false).  
It is a $n^C \times n^C$ matrix, with $n^C$ the number of classes.
````

```{figure} ../images/lec03_5_confusionmatrix.png
---
  name: lec03_5_confusionmatrix
  width: 100%
---
 . The confusion matrix for a binary classifier. <sub>Image from the author</sub>
 ```
The quantity in each cell $C_{i,j}$ corresponds to the number of observations known to be in group $i$ and predicted to be in group $j$.
The true cells are along the diagonal when $i=j$. Otherwise, if $i \neq j$, it is false. For $n^C =2$ there are two ways to be right, two ways to be wrong. The counts are called:
* $C_{0,0}$: true negatives
* $C_{1,1}$: true positives
* $C_{0,1}$: false positives
* $C_{1,0}$: false negatives

Here is a rephrase in the context of event classification: signal (1) versus background (0).

__True positives__  
We predict signal and the event is signal.

__True negatives__  
We predict background and the event is background.

__False positives__  
We predict signal but the event is background: our signal samples will have background contamination.

__False negatives__  
We predict background but the event is signal: we have signal contamination in the background but most importantly: we missed a rare signal event!

The false positive and false negatives misclassifications are also referred to as type I and type II errors respectively. There are usually phrased using statistical jargon of null hypothesis (background) and alternative hypothesis (signal). The definitions below merge the statistical phrasing with our context above:


````{prf:definition}
:label: 
__Type I error - False Positive__  
Error of rejecting the null hypothesis (background classified as signal) when the null hypothesis is true.

__Type II error - False Negative__  
Error of accepting the null hypothesis (signal not seen as signal) when the alternative hypothesis is actually true.
````

The type I error leads to signal samples not pure, as contaminated with background. But a type II error's is a miss on a possible discovery! Or in medical diagnosis, it can be equivalent to state "you are not ill" to a sick patient. Type II errors are in some cases and contexts much worse than type I errors.

## Performance measures

### For regression algorithms
There are many metrics used to evaluate the performance of regression algorithms, each with theirs pros and cons. 

````{prf:definition}
:label: rmseDef
The __root mean squared error (RMSE)__ is the square root of the mean squared error:
\begin{equation}
\text{RMSE} = \sqrt{ \frac{\sum_{i=1}^N (y^\text{pred} - y^\text{obs} )^2 }{ N } }
\end{equation}
````
RMSE ranges from 0 to infinity. The lower the RMSE, the better. By taking the square root we have an error of the same unit that the target variable $y$ we want to predict.

You may have seen in a statistics course the coefficient of determination, called $R^2$ or $r^2$. This is not really a measure of model performance, although it can be used as a proxy. What $r^2$ does is to measure of the amount of variance explained by the model. It is more a detector of variance than a performance assessment. Ranging from 0 to 1, with 1 being ideal. 


### For classification algorithms

The total model error, i.e. the sum of all wrong predictions divided by the total number of predictions, is not a good metric as it mixes types I and II errors. There are other error measurements more appropriate to measure the performance for classification. The more popular ones associated with machine learning are defined below:

````{prf:definition}
:label: errormetricsclassdef

__Accuracy__  
Rate at which the model is able to predict the correct values of both classes.
\begin{equation}
\text{Accuracy} = \frac{\text{True predictions}}{\text{All predictions}} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
\end{equation}

__Precision, or Positive Predictive Value (PPV)__  
Measure of the fraction of true predictions among all __positive predictions__.
\begin{equation}
\text{Precision} = \frac{\text{True predictions}}{\text{All Positive Predictions}} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}
\end{equation}

__Recall, or True Positive Rate (TPR)__  
Measure of the fraction of true predictions among all __true observations__.
\begin{equation}
\text{Recall} = \frac{\text{True predictions}}{\text{Actual Positive}} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}
\end{equation}

__F-Score, or F1__  describes the balance between Precision and Recall. It is the harmonic mean of the two:
\begin{equation}
\text{F1} =2 \; \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\end{equation}
````

The F-Score is a single metric favouring classifiers with similar Precision and Recall. But in some contexts, it is preferrable to favour a model with either high precision and low recall, or vice versa. There is a known trade-off between precision and recall. 

```{admonition} Exercise
:class: seealso
Find different examples of classification in which:
* a low recall but high precision is preferrable
* a low precision but high recall is preferrable 
```

## Bias and Variance
### Definitions
The generalization error can be expressed as a sum of three errors:
```{math}
:label:
\text{Total error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible}
```
The two first are _reducible_. In fact, we will see in the following how to reduce them as much as possible! The last one is due to the fact data is noisy itself. It can be minimized during data cleaning by removing outliers (or more upfront by improving the detector or device that collected the data). 

```{figure} ../images/lec03_5_bias_var_ierrors.png
---
  name: lec03_5_bias_var_ierrors
  width: 90%
---
 . Decomposition of the generalized error into the bias, variance and irreducible errors.  
 <sub>Image: [towardsdatascience.com](https://towardsdatascience.com/the-bias-variance-tradeoff-8818f41e39e9)</sub>
```
Increasing the model complexity will reduce the bias but increase the variance. Reversely, simplifying a model to mitigate the variance comes at a risk of a higher bias. In the end, the lowest total error is a trade-off between bias and variance.

````{prf:definition}
:label: biasdef
The __bias__ is an error coming from wrong assumptions on the model. 

A highly biased model will most likely underfit the data.

````
The bias implies not grasping the full complexity of the situation (think of a biased person making an irrelevant or indecent remark in a conversation).


````{prf:definition}
:label: variancedef
The __variance__ is an error stemming from the model's unreasonable sensitivity to fluctuations from the training data set.

A model with high variance is likely to overfit the data.

````
As its name suggest, a model incorporating fluctuations in its design will change, aka _vary_, as soon as it is presented with new data (fluctuating differently).

Below is a good visualization of the two tendencies for both regression and classification:

```{figure} ../images/lec03_5_underoverfit_reg_class_table.jpg
---
  name: lec03_5_underoverfit_reg_class_table
  width: 100%
---
 . Illustration of the bias and variance trade-off for regression and classification.  
 <sub>Image: LinkedIn Machine Learning India</sub>
```
Before learning on ways to cope with either bias or variance, we need first to assess the situation. How to know if our model has high bias or high variance?

### Identify the case
By plotting the cost function $J(\theta)$ with respect to the model's complexity. Increasing complexity can be done by adding more features, higher degree polynomial terms, etc. This implies running the training and validation each time with a different model to collect enough points to make such a graph:

```{figure} ../images/lec03_5_bias-variance-train-val-complexity.png
---
  name: lec03_5_bias-variance-train-val-complexity
  width: 90%
---
 . Visualization of the error (cost function) with respect to the model's complexity for the training and validation sets. The ideal complexity is in the middle region where both the training and validation errors are low and close to one another.  
 <sub>Image: [David Ziganto](https://dziganto.github.io/cross-validation/data%20science/machine%20learning/model%20tuning/python/Model-Tuning-with-Validation-and-Cross-Validation/)</sub>   
 ```

It can impractical to test several models with higher complexity. More achievable graphs would be to plot the error (cost function) with respect to the sample size $m$ or the number of epochs $N$:

```{figure} ../images/lec03_5_low_high_bias.webp
---
  name: lec03_5_low_high_bias
  width: 100%
---
```
```{figure} ../images/lec03_5_low_high_var.webp
---
  name: lec03_5_low_high_var
  width: 100%
---
.  Interpretation of the error plots as a function of the number of samples in the dataset for low and high bias/variance situations.
<sub>Images: [dataquest.io](https://www.dataquest.io/blog/learning-curves-machine-learning/)</sub>
```

The presence of a small gap between the train and test errors could appear like a good thing. But it important to quantify the training error and relate it to the desired accuracy: if the error is much higher than the irreducible error, chances are the algorithm is suffering from a high bias. 

The variance is usually spotted by the presence of a significant gap pertaining even if the dataset size $m$ increases, yet closing itself for large $m$ (hint for the following section on to cope with variance: getting more data). 


### How to deal with bias or variance

The actions to perform to mitigate either bias or variance once we have diagnosed the situation can be done on the dataset, on the model itself and on the regularization. The table below summarizes the relevant treatments to further optimize your machine learning algorithm in the good direction.

```{list-table}
:header-rows: 1

* - Action categories
  - Reducing Bias
  - Reducing Variance
* - On the dataset
  - 
  - Adding more (cleaned) data
* - On the model
  - Adding new features and/or polynomial features
  - Reducing the number of features
* - Regularization
  - Decreasing parameter $\lambda$
  - Increasing parameter $\lambda$
```
  

The tutorials will offer a good training for you (and validation ;) ) to diagnose and correctly optimize your machine learning algorithm. 

```{admonition} Learn more
:class: seealso
* [Confusion Matrix, Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) 
* [Machine Learning Model Performance and Error Analysis, LinkedIn](https://www.linkedin.com/pulse/machine-learning-model-performance-error-analysis-payam-mokhtarian)

```