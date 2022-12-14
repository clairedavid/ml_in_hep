# LHC Event Classification with Trees

```{admonition} Learning Objectives
:class: tip

In this tutorial, you will learn how to classify collisions from the Large Hadron Collider using decision trees and random forests. 

Main goals:
* __Implement a decision tree__
* __Compute by hand performance metrics__
* __Compare decision trees by changing hyperparameters__
* __Implement a random forest classifier__
* __Visualize the decision surface__
* __Plot a ROC curve and compare different classifiers' performance__

What will also be covered:
* How to load and explore a dataset
* How to plot with Matplotlib
* How to define custom functions
* How to debug a code
```

Let’s now open a fresh Jupyter Notebook and follow along!

<h2>Introduction</h2>

<h3>Higgs boson production modes</h3>

In particle physics, the Higgs boson plays an essential role, in particular (pun intended) it gives massive particles their observed mass. The Higgs boson can be produced in different ways - we call this "Higgs boson production mechanism." The main two production processes are: 
* gluon-gluon Fusion (ggF): two gluons, one from each of the incoming LHC protons, interact or “fuse” to create a Higgs boson.
* Vector Boson Fusion (VBF): a quark from each of the incoming LHC protons radiates off a heavy vector boson ($W$ or $Z$). These bosons interact or “fuse” to produce a Higgs boson.

```{figure} ../images/tuto_02_1_higgsfeyn.png
---
  name: tuto_02_1_higgsfeyn
  width: 90%
---
 .  Feynman diagrams for the gluon-gluon Fusion (ggF) process on the left and Vector Boson Fusion (VBF) on the right.  
 <sub>Image: ATLAS, CERN</sub>
```

The latter process, VBF, is very interesting to study as it probes the coupling between the Higgs boson and the two other vector bosons. This is seen on the Feynman diagram with the vertex between the two departing wavy branches of each vector boson V and the dashed line H representing the Higgs boson. Such configuration is said to be "sensitive to new physics", because there can be processes that are not part of the current theory, the Standard Model, arising there. Hence the importance to measure the rates of VBF collisions (how frequent does it happen). But before, how to select the Higgs boson VBF production from the other one, gluon-gluon Fusion?  
<center>__This is your mission!__  <sub>You will be guided, don't worry.</sub></center>  

<h3>Inside the Data</h3>  

````{margin}
Curious about CERN ATLAS Open Data initiative? Explore it more [here](https://atlas.cern/Resources/Opendata).
````
This tutorial will use ATLAS Open Data, which provides open access to proton-proton collision data at the LHC for educational purposes. 
 
In the VBF process, the initial quarks that first radiated the vector bosons are deflected only slightly and travel roughly along their initial directions. They are then detected as particle "jets" in the different hemispheres of the detector. Jets are reconstructed as objects. Although they are more of a conical shape, they are stored in the data as a four-vector entity, with a norm, two angles and an energy. 

The collisions have been filtered to select those containing each a Higgs boson, four leptons and at least two jets. 

We will focus on two variable for now: 
* $|\Delta\eta_{jj}|$: it corresponds to the angle between the two jets ($\eta$ is the pseudorapidity)
*   $m_{jj}$:   the invariant mass of the two jets
 
These variable are already calculated in the data samples.

## Explore the Data
### Get the Data 
The datasets can be found [here](https://drive.google.com/drive/folders/1b_GDA2bfsUhlvzX-A7RjHoHCL5Z8-bkb?usp=sharing). Download the files and put them in your GDrive.

To load the data on Google Colab, you will need to run a cell with these lines:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
Then using `!ls` you can locate the files or folder.

Before playing with the data, let's import libraries.

```python
import os, sys
import pandas as pd
import numpy as np

# set a seed to ensure reproducibility
seed = 42
rnd  = np.random.RandomState(seed)

# Matplotlib plotting settings
import matplotlib as mp
import matplotlib.pyplot as plt
%matplotlib inline
print('matplotlib version: {}'.format(mp.__version__))

FONTSIZE = 16
params = {
         'axes.labelsize': FONTSIZE,
         'axes.titlesize': FONTSIZE,
         'xtick.labelsize':FONTSIZE,
         'ytick.labelsize':FONTSIZE}
plt.rcParams.update(params)
```
__Question 1.0: Get the data__  
Use `pd.read_csv` to store each dataset into a dataframe. Name them `train`, `valid` and `test` respectively. 
Explore the variables by printing the first five rows.

The `sample` column stores the labels of the collisions: +1 corresponds to VFB and -1 to ggF.

__Question 1.1: Warm-up: inspect the data. How many events (rows) does each file contain?__  
__Question 1.2: How many events of each process (VFB and ggF) does each file contain?__  

Ask for hint(s) to the instructor if you are stuck.

### Visualize the Data
Let's draw a scatter plot to see how the data look like! But first, we will create reduced dataset with only the necessary variables. Copy the following in your notebook:
```python
# GLOBAL VARIABLES
XNAME = 'detajj'; XLABEL = r'$|\Delta\eta_{jj}|$'
YNAME = 'massjj'; YLABEL = r'$m_{jj}$ (GeV)'

inputs= [XNAME, YNAME] ;

XBINS = 5 ; XMIN = 0 ; XMAX = 5    ; XSTEP = 1
YBINS = 5 ; YMIN = 0 ; YMAX = 1000 ; YSTEP = 200

# Creating reduced datasets with detajj & massjj only
X_train = train[inputs] ; y_train = train['sample']
X_valid = valid[inputs] ; y_valid = valid['sample']
X_test  =  test[inputs] ; y_test  =  test['sample']
```
The plotting macro is given, but you will have to modify it later:
```python
def plot_scatter(sig, bkg, 
              xname=XNAME, xlabel=XLABEL, xmin=XMIN, xmax=XMAX, xstep=XSTEP,
              yname=YNAME, ylabel=YLABEL, ymin=YMIN, ymax=YMAX, ystep=YSTEP,
              fgsize=(6, 6), ftsize=FONTSIZE, alpha=0.3, title="Scatter plot"):
  
  fig, ax = plt.subplots(figsize=fgsize)

  # Annotate x-axis
  ax.set_xlim(xmin, xmax)
  ax.set_xlabel(xlabel)
  ax.set_xticks(np.arange(xmin, xmax+xstep, xstep))

  # Annotate y-axis
  ax.set_ylim(ymin, ymax)
  ax.set_ylabel(ylabel)
  ax.set_yticks(np.arange(ymin, ymax+ystep, ystep))

  # Scatter signal and background:
  ax.scatter(sig[xname], sig[yname], marker='o', s=15, c='b', alpha=alpha, label='VBF')
  ax.scatter(bkg[xname], bkg[yname], marker='*', s= 5, c='r', alpha=alpha, label='ggf')

  # Legend and plot:
  ax.legend(fontsize=ftsize, bbox_to_anchor=(1.04, 0.5), loc="center left", frameon=False) 
  ax.set_title(title, pad=20)
  plt.show()
```

__Question 1.3: Make a scatter plot of the training data__ , with $|\Delta\eta_{jj}|$ on the $x$-axis and $m_{jj}$ on the $y$-axis. You will have to split the data sample in signal (VBF) and background (ggF).

## Decision Tree
Let's use Scikit-Learn to make a first shallow decision tree.

```python
from sklearn import tree
from sklearn.tree import export_text
```
__Question 2.1: Make a decision tree with a maximum depth of 2.__  
__Question 2.2: Plot the tree using `tree.plot_tree` command, with the `filled` option activated.__  

You will see something like this:


```{figure} ../images/tuto_02_2_shallowtree.png
---
  name: tuto_02_2_shallowtree
  width: 80%
---
 .  Representation of the Decision Tree.
 <sub>Image: from Scikit-Learn `tree` library</sub>
```
__Question 2.3: Comment the tree.__  
Describe what is the representation. What are each variable? Which direction (left/right) is true/false? What are the nodes' colouring meaning? Where goes the signal, where goes the background? Which leaves are the purest? For which category?

__Question 2.4: Calculate the accuracy from the numbers displayed in the leaves.__  
Detail your calculations.


## Performance metrics
We will compute functions to evaluate the tree and compare with Scikit-Learn predefined methods. First, let's get a Confusion Matrix from Scikit-Learn. Let's first import the library:
```python
from sklearn import metrics
```

The way the confusion matrix is called is:
```
cm = metrics.confusion_matrix(y_obs, y_preds)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```
__Questions 3.1: Confusion Matrix__  
Using the `predict()` method on your classifier, write the code to show the confusion matrix.

__Question 3.2: Comment it__  
How is the confusion matrix encoded in Scikit-Learn? Is it the same as in the lecture? Find and explain in which cells are the True Positives (TP), True Negatives (TN), False Positive (FP) and False Negatives (FN). 






