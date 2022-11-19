# Hyperparameters in DL
````{margin}
* We will see in Lecture 13 how to choose the most relevant optimization algorithm. For now assume we have the best model (boosted tree or neural networ etc).
````
For a given machine learning model<sup>*</sup>, there are usually several hyperparameters configuring it. Tweaking their values to reach the optimum performance of the model is what is referred to as __hyperparameter tuning__. This could be done manually, using ranges of possible values for each parameter and embedded `for` loops to go over all possible combinations. But it would be very tedious work and impractical as there may be too many possibilities - and recall that a single training involves numerous computations!

```{admonition} Exercise
:class: seealso
If your model has five hyperparameters and you want to try 10 different values for each of them, how many tuning combinations will there be?
```

## GridSearchCV
Here comes a great method called Grid Search. This process has been coded as a tool available in python libraries such as `Keras` and `PyTorch`, as we will soon see. It is sometimes written `GridSearchCV`, where the CV suffix stands for Cross-Validation.

````{prf:definition}
:label: 
__Grid Search__ is an exhaustive scan of the hyperparameters from manually specificed values in order to find the combination maximizing the model's performance.
````


 


```{admonition} Learn More
:class: seealso
Yoshua Bengio, "Practical Recommendations for Gradient-Based Training of Deep Architectures" (2012) [arXiv:1206.5533](https://arxiv.org/abs/1206.5533)
```