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
````{margin}
```{warning}
Caution: the path in the code example may not match your local folder. You may have to adapt the command with your file's location.
```
````
Often you will not know how data is presented in a file. In bash, the `cat` command will display the entire content of a text file. But caution: your file can have thousands of lines! What is usually relevant to see is e.g. if the file has a header row. The command that displays the firt rows is `head`. By default it will show 10 rows, but you can tweak this number with the `-n` option:

```bash
!head -n tutorial_1.csv
```
You will see something like this (perhaps not the exact numbers, but it is not important):
```bash
x,y
5.4881350392732475,29.65450786127179
7.151893663724195,34.99358978951999
6.027633760716439,35.99427342345802
5.448831829968968,24.815775427356154
```

Too see how many lines your file has, use the "word count" command `wc` with the `-l` option for line:
```bash
! wc -l tutorial_1.csv  

51 tutorial_1.csv
```

This is useful to know which amount of data we are dealing with.  
And we also see the file has a header row. Let's now load its content in a pythonic way.

### Loading csv content into a dataframe
A dataframe is a data structure from the pandas software library designed for tabular data. It can be compared to an array with rows and columns that can be modified in an intuitive way for the programmer. Pandas allows importing data from various file formats. There is a special method for loading the content from a `.csv` file. Let's first do the proper imports:

```python
filename = "tutorial_1.csv"
df = pd.read_csv(filename)
df.head()
```


## Step 2: Visualizing the Data
Let's make a plot! For this, we will use the [Matplotlib library](https://matplotlib.org/). Their documentation is structured with lots of [examples](https://matplotlib.org/stable/plot_types/index.html) to get inspiration from. Let's make now a scatter plot of the data:


