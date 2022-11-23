# Let's train our NN!

Time to gather all the notions covered in this lecture and learn how to build a deep learning model.

## Steps in Building your NN
Designing a machine learning algorithm has a particular workflow, usually called life-cycle (but workflow would be more accurate here). 

The usual steps are:

1. Get the Data
1. Visualize the Data
1. Prepare the Data
1. Choose the Model
1. Train the Model
1. Tune the Model
1. Evaluate the Model
1. Make Predictions

These steps may be coinced in a different way in industry, with e.g. the last one called "deployment". We will stay in the academic realm with prediction making, as this is all that is about.

But the most important step is missing. It's the very first one:

### 0. Frame the problem

__The Big Picture__  
Before even starting to code anything, it is crucial to get a big picture on the challenge and ask oneself: what is the objective? What exactly do I want to predict?  

__What is before and after__  
Framing implies to think of what comes before and after the optimization procedure. The learning algorithm to build is likely to insert itself into an analysis or quantitative study. Documenting oneself on what is done before, likely the data taking procedure, is important to gather... more data on the data. Can the dataset be trusted, partially or entirely? Same regarding what comes after the predictions. Are these predictions final? Or rather, are the outputs become the inputs to another study? Thinking of the inputs and outputs can provide already a good guidance on how you may solve the problem. It could even drastically change the way you may proceed. In a data analysis involving a BDT (Boosted Decision Trees), it was found that an increase in performance of some percent would be absorbed at the next stage of the analysis during the test statistics, where different portions of the data had large uncertainties associated with them.  

__How would solution(s) look like__  
The next investigation is on the solution(s). Perhaps previous attempts in the past have been done to solve the problem. Or solutions exist but they are not reaching the desired precision. In this case it is always a good idea to collect and read some papers to get the achieved ballparks regarding accuracy, sensitivity, specificity. If solutions are inexistant, it is still possible to think of the consequences of the possible solution(s). Will it bring novelty into the field?

__How to evaluate the performance__  
The next step is to think of the proper metrics to evaluate your future solution. This is a hard step, yet crucial to ... 

__Which type of ML is it__  
Anticipating Step 3, 
Then the model (anticipating Step 3). What type of Machine Learning it is? Is it regression or classification? 

### 1. Visualize the Data
Before even starting to prepare the data for machine learning purposes, it is recommended to see how the data look like.  

The data can be big, so it is cautious to first know the number of columns (features) and rows (data instances). 

```python
# Counting the number of rows
nb_rows = len(df.index)
# Or:
nb_rows = df.shape[0]

# Columns
nb_cols = len(df.columns)
# Or
nb_cols = df.shape[1]

# To list the columns:
print(df.columns)

```

Or more directly:
```python
df.info()
```
which will show the types for each column and also the memory usage.

Dataframes in Jupyter-Notebook neatly display as a human readable table with the columns highlighted and rows indexed. As the dataset can be big, you can print only the 5 first rows:
```python
df.head(5)
```
This will work on Jupyter-Notebook but in a regular python script, you may need to insert it into a print statement such as `print(df.head(5)`. If the data is sorted, you may not have a correct glimpse of values. For instance if the signal is first, the target column $y$ would display -1 (names $y$ and the value of 1 can also change, you will have to check that). Once you check the number of instances, you can display several rows picked randomly. If you have 10,000 instances, you can explore:
```python
df.iloc[ [0, 5000, 9000] , : ]
```
This would show you three instanced at the start, middle and end of the dataset.

It is also good to check how balanced your dataset is in terms of signal vs background samples. 
```

```

Balanced dataset ? 
Matplotlib
seaborn 


### 2. Prepare the Data

### 3. Choose the Model

### 4. Train the Model

### 5. Tune the Model

### 6. Evaluate the Model






## What is PyTorch?
One can code a big neural network from scratch in python, declaring all the functions, classes etc... That would be very tedious and likely not computationally optimized for speed. Most importantly: it's been already done. There are indeed dedicated libraries for designing and developing neural networks and deep learning technology. 

### ML frameworks
The two more powerful and popular open-source machine learning frameworks are Keras and PyTorch. They are used by both researchers and developers because they provide fast and flexible implementation. While Keras is more readable and concise with its simple architecture, it is slower in comparison with PyTorch, thus more suited for small datasets. Pytorch is developed and maintained by Facebook. It is built to use the power of GPUs for faster training and is deeply integrated into python, making it easy to get started.

### Tensors




Automatic differentiation -> AutoGrad
