# Hyperparameters in DL
````{margin}
$*$ We will see in Lecture 13 how to choose the most relevant optimization algorithm. For now assume we have the best model (boosted tree or neural networ etc).
````
For a given machine learning model<sup>*</sup>, there are usually several hyperparameters configuring it. Tweaking their values to reach the optimum performance of the model is what is referred to as __hyperparameter tuning__. This could be done manually, using ranges of possible values for each parameter and embedded `for` loops to go over all possible combinations. But it would be very tedious work and impractical as there may be too many possibilities - and recall that a single training involves numerous computations!

```{admonition} Exercise
:class: seealso
If your model has five hyperparameters and you want to try 10 different values for each of them, how many tuning combinations will there be?
```

Luckily, some tools are available to do the tuning!

## Grid Search
Here comes a great method called Grid Search. This process has been coded as a tool available in python libraries such as Scikit-Learn and PyTorch, as we will soon see. 

### Definition

````{prf:definition}
:label: 
__Grid Search__ is an exhaustive scan of the hyperparameters from manually specificed values in order to find the combination maximizing the model's performance.
````

The Grid Search method is also called _parameter sweep_ (although it should be called _hyperparameter sweep_). It is what it does: it sweeps over all possibilities of the hyperparameters the user provides. 

__How does Grid Search knows what is best?__  
The guidance here is a performance assessment, performed using a $k$-fold cross-validation. This is why most Grid Search modules in common libraries are named `GridSearchCV`, where the CV suffix stands for cross-validation. Which performance metric is used? This can be entered by the user; with proper thinking first on which performance metric is the most relevant for the task at hand! The default in Scikit-Learn is the accuracy for classification and $r^2$ for regression.  

### GridSearchCV
Below is an implementation of Grid Search in Scikit-Learn, using one of Scikit-Learn dataset (housing price estimation).





```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
```

The `param_grid` is a python dictionnary with the hyperparameters as keys. For each hypermeter, a list of values should be provided. 

### Limitations



## Random Search





## Bayesian Optimization

 


```{admonition} Learn More
:class: seealso
Yoshua Bengio, "Practical Recommendations for Gradient-Based Training of Deep Architectures" (2012) [arXiv:1206.5533](https://arxiv.org/abs/1206.5533)
```