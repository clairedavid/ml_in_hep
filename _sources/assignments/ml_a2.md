# Assignment 2: Decision Trees

````{margin}
```{admonition} Dataset
Find the repository on GDrive [here](https://drive.google.com/drive/folders/1b_GDA2bfsUhlvzX-A7RjHoHCL5Z8-bkb?usp=sharing).
```
````
## 1. Decision Stump by hand
Using the data from tutorial 2, you will implement a one-level decision tree, or decision stump.  
You will use the CART algorithm (Classification And Regression Tree). Recall the Gini's index measuring the impurity is defined as:  
\begin{equation*}
G_i = 1 - \sum_{k=1}^{N_\text{classes}} \left( \frac{N_{k, i}}{ N_i} \right)^2 
\end{equation*}
The cost function is
\begin{equation*}
J(k, t_k) = \frac{n_\text{left}}{n_\text{node}} G_\text{left} + \frac{n_\text{right}}{n_\text{node}} G_\text{right} \;,
\end{equation*}
where $k$ is a given feature and $t_k$ the threshold on that feature. We will use $|\Delta\eta_{jj}|$ and $m_{jj}$ as our two input variables. The main function `decision_stumper` should return the optimized threshold and cost function values for a given feature. It should take as arguments:
* the dataframe
* the variable name of the input feature 
* the class name (column name where labels are stored)
* the class values (in an array)
* the numbers of threshold values swiping the interval of the feature

__1.1: Get and load the data__  
Get the `train` dataset and load the relevant columns in a dataframe.

__1.2 Compute the Gini index__    
Write a function computing the Gini index value. Make your code as general as possible.  
Add in the next cell a series of tests.  
_Bonus: secure your code to prevent a division by zero._

__1.3 Calculate the cost__  
Write a function computing the cost function in the CART algorithm. 

__1.4 Main function: code a Decision Stump__  
Write the main function `decision_stumper` that will call the functions defined above. Call your function on each input feature and conclude on the final cut for your decision stump.

You just coded a decision stump by hand!

````{margin}
```{tip}
If you cannot complete the previous question, choose a feature and a threshold yourself to be able to draw the decision boundary.
```
````
__1.5 Plot the cut__  
Use the `plot_scatter` function from the second tutorial and modify it to draw the line corresponding to the optimized threshold from the decision stump. You can use Matplotlib's `axhline` or `axvline` method for drawing a horizontal or vertical line respectively. Try to be as general as possible in the input arguments.  
_Hints provided on demand during office hours._

## 2. Plotting mission: the overtraining check
The goal of this exercise is to understand and reproduce the following plot:


