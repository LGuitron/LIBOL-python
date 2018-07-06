# LIBOL Python

Library for Online Learning algorithms in Python 3
This project is based on LIBOL https://github.com/LIBOL/LIBOL by stevenhoi,
and contains some First Order and Second Order algorithms, as well as options for 
customizing these algorithms as needed (regularization, kernel trick, hyperparameters, ...)

## Getting Started

Clone the repository and install the dependencies listed in the file requirements.txt, these dependencies are:
```
numpy
oct2py
nose2
```

## Run Test

To verify the correct installation of the Library execute the following command:
```
python run.py
```
This command runs the Perceptron algorithm for the Breast Cancer Scale dataset from LIBSVM datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.</br>
The results of this execution should be somethin like this:

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

        Binary Classification  | Multiclass Classification
        ---------------------- | -------------
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
python run.py -h
usage: run.py [-h] [-t TASK] [-a ALGORITHM] [-d PATH_TO_DATASET]
              [-f FILE_FORMAT]

Online learning algorithms selection

optional arguments:
  -h, --help          show this help message and exit
  -t TASK             Classification task (Default bc): {bc, mc}
  -a ALGORITHM        OL algorithm to run (Default Perceptron):{Perceptron,
                      PA, PA1, PA2, OGD,SOP, CW, SCW, SCW2, AROW}
  -d PATH_TO_DATASET  Path to dataset (Default
                      ./data/test/bc/breast_cancer_scale.txt)
  -f FILE_FORMAT      Input file format (Default libsvm)


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



## Hyperparameter Selection

Hyperparameters for many of the algorithms can be setup in the options.py file.
The variables that can be modified are:






## Compare Algorithms and generate Plot





