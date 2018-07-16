# LIBOL Python

Library for Online Learning algorithms in Python 3
This project is based on LIBOL https://github.com/LIBOL/LIBOL by stevenhoi,
and contains some First Order and Second Order algorithms, as well as options for 
customizing these algorithms as needed

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
-a : Algorithm to run, possible options:
```

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

```
 -d : Path to training dataset

-t : Type of problem to solve, possible options are:
     bc - Binary Classification
     mc - Multiclass Classification
 
 -f : File format of input data, formats supported are:
     libsvm

 -h / --help: Adding this flag displays input information in the command line

```
## Examples

```
python run.py -a OGD -t bc -d ./data/bc/a7a.t -f libsvm
-------------------------------------------------------------------------------
Dataset name:  ./data/bc/a7a.t ( n= 16461  d= 123  m= 2 )        nb of runs (permutations):  20
-------------------------------------------------------------------------------
Algorithm:  OGD
mistake rate:  0.2001 +/- 0.0021
nb of updates:  6962.85 +/- 143.0819
cpu time (seconds):  0.7073 +/- 0.0024
-------------------------------------------------------------------------------

python run.py -a M_PA2 -t mc -d ./data/mc/mnist -f libsvm
-------------------------------------------------------------------------------
Dataset name:  ./data/mc/mnist ( n= 60000  d= 780  m= 10 )       nb of runs (permutations):  20
-------------------------------------------------------------------------------
Algorithm:  M_PA2
mistake rate:  0.1444 +/- 0.0009
nb of updates:  23302.65 +/- 77.0086
cpu time (seconds):  1.913 +/- 0.0531
-------------------------------------------------------------------------------

```
## Test performance vs LIBOL Matlab

This library is based on LIBOL by stevenhoi.
For this reason unit tests are provided with the puprose of testing the following:
* Similar mistake rates when executing algorithms with the same values for hyperparameters
* Equal or better running time for this library compared to LIBOL Matlab

To run the tests execute the command
```
nose2
```
This tests run all of the algorithms for both Matlab and Python implementations with the exception of the Kernel Perceptron and Kernel OGD. The files to be tested for binary classification and multiclass classification are located in data/test/bc, and data/test/mc respectively, additional files can be added into these directories for additional testing.

## Hyperparameter Selection

Hyperparameters for many of the algorithms can be setup in the options.py file.
Modify this file in order to pick the parameters that work best for a particular dataset.
The values of the hyperparameters have to be setup per algorithm, and are the following:

Parameter   | Type        |                 Description
------------|-------------|------------------------------------------------------------
bias        | Boolean     | Add bias weight to w
C           | Float       | Learning Rate / Aggresiveness
loss_type   | Integer     | Loss function in OGD (0: 0-1 loss, 1: Hinge, 2: Log, 3: Square )
a           | Float       | Initial value in diagonal matrix for prediction <br> confidence in CW algorithms
eta         | Probability | Confidence threshold for CW algorithms
regularizer | Function    | Function for sparcity regularization (functions in regularizers/Regularizer.py)
max_sv      | Integer     | Support vectors taken into account in kernel algorithms
sigma       | Float       | Variance parameterfor Gaussian Kernel
kernel      | Function    | Kernel method used (gaussian_kernel implemented, <br> additional kernels can be included in kernels/Kernels.py)  

### Binary Classification

   Parameter   | Perceptron | PA | PA1 | PA2 | OGD | CW | AROW | NAROW | SOP | SCW | SCW2|  Kernel Perceptron | Kernel OGD         
 --------------|------------|----|-----|-----|-----|----|------|-------|-----|-----|-----|--------------------|------------ 
   bias        |      X     | X  |  X  |  X  |  X  |  X |   X  |   X   |  X  |  X  |  X  |                    |
   C           |            |    |  X  |  X  |  X  |    |   X  |   X   |     |  X  |  X  |                    |      X     
   loss_type   |            |    |     |     |  X  |    |      |       |     |     |     |                    |      X
   a           |            |    |     |     |     |  X |   X  |   X   |  X  |  X  |  X  |                    |
   eta         |            |    |     |     |     |  X |      |       |     |  X  |  X  |                    |
   regularizer |            |    |     |     |  X  |    |      |       |     |     |     |                    |
   max_sv      |            |    |     |     |     |    |      |       |     |     |     |         X          |      X
   sigma       |            |    |     |     |     |    |      |       |     |     |     |         X          |      X
   kernel      |            |    |     |     |     |    |      |       |     |     |     |         X          |      X 
   

### Multiclass Classification

   Parameter   | M_PerceptronM | M_PerceptronS | M_PerceptronU | M_OGD | M_PA | M_PA1 | M_PA2 | M_CW | M_SCW1 | M_SCW2 | M_AROW
 --------------|---------------|---------------|---------------|-------|------|-------|-------|------|--------|--------|-------
   bias        |       X       |        X      |       X       |   X   |   X  |   X   |   X   |   X  |    X   |    X   |    X 
   C           |               |               |               |   X   |   X  |   X   |   X   |      |    X   |    X   |    X
   a           |               |               |               |       |      |       |       |   X  |    X   |    X   |    X
   eta         |               |               |               |       |      |       |       |   X  |    X   |    X   |    
   regularizer |               |               |               |   X   |      |       |       |      |        |        |     


### Automatic hyperparameter adjustment

For all algorithms the initial value of the hyperparameters are set in the options.py file.
If specified, the execution of an algorithm can include a process of hyperparameter tuning in which all parameter values are kept fixed with the exception of one that takes different values from a specified range. After repeating this process with all hyperparameters the best values will be exported to a text file with the name "algorithm_dataset_bestparams.txt", when this process is completed the algorithm will be executed once more with the best values found.

The range of values to be tested for every hyperparameter is represented by lists in the options.py file, modifying these lists might improve the results obtained in some cases.

## Compare Algorithms and generate Plot





