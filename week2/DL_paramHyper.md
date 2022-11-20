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
:label: gridsearchdef
__Grid Search__ is an exhaustive scan of the hyperparameters from manually specificed values in order to find the combination maximizing the model's performance.
````

The Grid Search method is also called _parameter sweep_ (although it should be called _hyperparameter sweep_). It is what it does: it sweeps over all possibilities of the hyperparameters the user provides. 

__How does Grid Search knows what is best?__  
The guidance here is a performance assessment, usually using a $k$-fold cross-validation. This is why most Grid Search modules in common libraries are named `GridSearchCV`, where the CV suffix stands for cross-validation. Which performance metric is used? This can be entered by the user; with proper thinking first on which performance metric is the most relevant for the task at hand! The default in Scikit-Learn is the accuracy for classification and $r^2$ for regression.  

### GridSearchCV
Below is an implementation of Grid Search in Scikit-Learn.

```python
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [3, 10, 30], 
               'max_features': [2, 4, 6, 8],
               'bootstrap': [True, False]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, 
                           param_grid, 
                           cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X, y)
```
<sub>From Aurélien Géron, _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (Second Edition)</sub>

The `param_grid` is a python dictionnary with the hyperparameters as keys. For each hyperparameter, a list of values should be provided. 

````{tip}
It is recommended while deciding on which values to choose to opt for consecutive powers of 10. In the case of a classification model involving a learning rate $\alpha$, a good call would be:

```python
param_grid = {
    "alpha": np.power(10, np.arange(-2, 1, dtype=float))
}
```
This would return `[0.01  0.1   1.]` for the values of $\alpha$ to test.
````

The results of the search are stored in the attribute `cv_results_`.

```python
cvres = grid_search.cv_results_
```
This function can print the `mean_test_score` (according to the example above) along with the values for each hyperparameters:
```python
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

The best estimator is given by `grid_search.best_estimator_`. 

### Limitations
Grid Search has several drawbacks:
* As the search is limited to the hyperparameter values entered by the user, if the process returns the minimum or maximum value of a given hyperparameter range, it is likely the score will improve by extending the range.
* The search is time-consuming. All values must be assessed (and $k$ times with the cross-validation!) before ranking the combinations. It becomes impractical when the hyperparameter space becomes very large.

Here is when the Random Search comes into play.

## Randomized Search
In the Randomized Search, hyperparameters are not defined as discrete values but 

````{prf:definition}
:label: randomsearchdef
The __Randomized Search__ is a tuning method consisting of evaluating a given number of combinations 
````



```{admonition} Learn More
:class: seealso
Yoshua Bengio, "Practical Recommendations for Gradient-Based Training of Deep Architectures" (2012) [arXiv:1206.5533](https://arxiv.org/abs/1206.5533)

"Comparing randomized search and grid search for hyperparameter estimation" on [Scikit-Learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)

```