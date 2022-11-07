# Linear Regression in Python

## Learning Objectives
In this tutorial, you will learn how to code a linear regressor in python. The steps will be detailed and the present document will show you basic commands in python to get started. 

__Goals:__ 
* __Code a linear regressor to fit a dataset of one independent variable with its target__
* __Display visualization of the gradient descent__

What will also be covered:
* How to load and explore a dataset
* How to plot with Matplotlib
* How to define custom functions
* How to printout and store intermediary results
* How to structure and comment code

Let's now open a fresh Jupyter Notebook and follow along!

## Step 1: Retrieving and Exploring the Dataset
The first step is to get the dataset. It can be accessed [here](../data/tutorial_1.csv). We will see how to localize, explore and load it.

### Localizing the file
In Jupyter Notebook, it is possible to write bash commands (the language used in a console, or terminal) by appending a `!` at front of a code cell.

The `pwd` command  stands for "print working directory" and will show you were you are currently running your code. 
```bash
!pwd
```

To list files, use the `ls` command of `ls -l` to display more details:
```bash
!ls -l
```

### Exploring the file
Often you will not know how data is presented in a file. In bash, the `cat` command will display the content of a text file. But caution: your file can have thousands of lines! What is usually relevant to see before loading it is just the first rows. There is a command for that: it is `head`:
````{margin}
```{warning}
Caution
```
````
```bash
!head ../data/tutorial_1.csv
```
You will see something like this (perhaps not the exact numbers, but it is not important):


### Loading csv content into a dataframe
Dataframes are ... 


## Step 2: Visualizing the Data
Let's make our first plot! For this, we will use the [Matplotlib library](https://matplotlib.org/). Their documentation is structured with lots of [examples](https://matplotlib.org/stable/plot_types/index.html) to get inspiration from. Let's make now a scatter plot of the data:


