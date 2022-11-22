# Let's train our NN!

Time to gather all the notions covered in this lecture and learn how to build a deep learning model.

## Deep Learning Model Life-Cycle
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

0. __Frame the problem__

Before even starting to code anything, it is crucial to get a big picture on the challenge and ask oneself: what is the objective? What exactly do I want to predict?  

Framing implies to think of what comes before and after the optimization procedure. The learning algorithm to build is likely to insert itself into an analysis or quantitative study. Documenting oneself on what is done before, likely the data taking procedure, is important to gather... more data on the data. Can the dataset be trusted, partially or entirely? Regarding the output, it is essential to know how it will be done. Is it the end result of a paper? In that case it is recommended to have done some reading 


## What is PyTorch?
One can code a big neural network from scratch in python, declaring all the functions, classes etc... That would be very tedious and likely not computationally optimized for speed. Most importantly: it's been already done. There are indeed dedicated libraries for designing and developing neural networks and deep learning technology. 

The two more powerful and popular open-source machine learning frameworks are Keras and PyTorch. They are used by both researchers and developers because they provide fast and flexible implementation. While Keras is more readable and concise with its simple architecture, it is slower in comparison with PyTorch, thus more suited for small datasets. Pytorch is developed and maintained by Facebook. It is built to use the power of GPUs for faster training and is deeply integrated into python, making it easy to get started.

### Tensors




## Neural Network step-by-step with PyTorch





PyTorch is integrated into Python.

Tensors 
Automatic differentiation -> AutoGrad
