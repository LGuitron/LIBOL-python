# LIBOL Python

Library for Online Learning algorithms in Python 3
This project is based on LIBOL https://github.com/LIBOL/LIBOL by stevenhoi,
and contains some First Order and Second Order algorithms, as well as options for 
customizing these algorithms as needed

## Algorithms

Binary Classification  | Multiclass Classification
---------------------- | -------------------------
AROW                   | M_AROW
CW                     | M_CW
Kernel_OGD             | M_OGD
Kernel_Perceptron      | M_PA
NAROW                  | M_PA1
OGD                    | M_PA2
PA                     | M_PerceptronM
PA1                    | M_PerceptronS
PA2                    | M_PerceptronU
Perceptron             | M_SCW1
SCW                    | M_SCW2
SCW2                   | 
SOP                    |


## Getting Started

Clone the repository and install the dependencies listed in the file requirements.txt, these dependencies are:
```
numpy
oct2py
nose2
six
Octave
```

## Run Test

To verify the correct installation of the Library execute the following command:
```
python run.py
```
This command runs the Perceptron algorithm for the Breast Cancer Scale dataset from LIBSVM datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.</br>
The results of this execution should be something like this:

```
-------------------------------------------------------------------------------
Dataset name:  ./data/test/bc/breast_cancer_scale.txt ( n= 683  d= 10  m= 2 )    nb of runs (permutations):  20
-------------------------------------------------------------------------------
Algorithm:  Perceptron
mistake rate:  0.0582 +/- 0.004
nb of updates:  39.75 +/- 2.7363
cpu time (seconds):  0.0056 +/- 0.0001
-------------------------------------------------------------------------------
```

## Execution Parameters

The main execution of an online learning algorithm is made from the run.py file,
the execution of this file requires the following console parameters:

```
-a : Algorithm to run (see table above)

 -d : Path to training dataset

-t : Type of problem to solve, possible options are:
     bc - Binary Classification
     mc - Multiclass Classification
 
 -f : File format of input data, formats supported are:
     libsvm

 -h / --help: Adding this flag displays input information in the command line

```
### Input Examples

```
python run.py -a OGD -t bc -d ./data/bc/a7a.t -f libsvm
python run.py -a M_PA2 -t mc -d ./data/mc/mnist -f libsvm

```
### Output Example
```
-------------------------------------------------------------------------------
Dataset name:  ./data/bc/a7a.t ( n= 16461  d= 123  m= 2 )        nb of runs (permutations):  20
-------------------------------------------------------------------------------
Algorithm:  OGD
mistake rate:  0.2001 +/- 0.0021
nb of updates:  6962.85 +/- 143.0819
cpu time (seconds):  0.7073 +/- 0.0024
-------------------------------------------------------------------------------
```
## Tests

Tests are included in this library with the purpose of comparing execution time and performance of algorithms with the ones implemented in LIBOL by stevenhoi.
These tests also allow quick verification of the functionality of elements in the library.

To run the tests execute the command
```
nose2
```

## Hyperparameter Selection

Hyperparameters for many of the algorithms can be setup in the options.py file.
Modify this file in order to pick the parameters that work best for a particular dataset.

Parameter   | Type        |                 Description
------------|-------------|------------------------------------------------------------
bias        | Boolean     | Add bias weight to w
C           | Float       | Learning Rate / Aggresiveness
loss_type   | Integer     | Loss function in OGD (0: 0-1 loss, 1: Hinge, 2: Log, 3: Square )
a           | Float       | Initial value in diagonal matrix for prediction <br> confidence in CW algorithms
eta         | Probability | Confidence threshold for CW algorithms
regularizer | Function    | Function for sparcity regularization (functions in regularizers/Regularizer.py)
max_sv      | Integer     | Number of support vectors taken into account in kernel algorithms
sigma       | Float       | Variance parameter for Gaussian Kernel
kernel      | Function    | Kernel method used (gaussian_kernel implemented, <br> additional kernels can be included in kernels/Kernels.py) 

### Automatic hyperparameter tuning

For all algorithms the initial value of the hyperparameters are set in the options.py file.
If specified, the execution of an algorithm can include a process of hyperparameter tuning in which all parameter values are kept fixed with the exception of one that takes different values from a specified range. After doing this process with all hyperparameters the algorithm will be executed once more with the best values found.

The ranges of values to be tested for every hyperparameter are represented by lists in the options.py file, modify these lists o use any desired ranges of values.

## Compare Algorithms and generate Plot

Algorithms are compared based on the task that they are able to perform (binary classification or multiclass classification)
The console parameters that can be set for this comparison are:

 -d : Path to training dataset

-t : Type of problem to solve, possible options are:
     bc - Binary Classification
     mc - Multiclass Classification
 
 -f : File format of input data, formats supported are:
     libsvm

### Input Examples
```
python compare.py -t bc -d ./data/bc/a7a.t -f libsvm
python compare.py -t mc -d ./data/mc/mnist -f libsvm
```
The algorithms will be executed with the parameter values specified in init_options.py, and the ranges of values specified in CV_algorithm.py
Running the comparison will display algorithm statistics for all algorithms for the selected task (bc or mc), as well as their corresponding plots comparing their performance in terms of error rate, number of updates, and time of computation.

### Output Example
#### Binary Classification
![alt text](https://github.com/LGuitron/LIBOL-python/blob/master/results/bc_plot_error_rate.png)

#### Multiclass Classification
![alt text](https://github.com/LGuitron/LIBOL-python/blob/master/results/mc_plot_error_rate.png)
