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
This command runs the Perceptron algorithm for the Breast Cancer Scale dataset from LIBSMV datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.</br>
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

### Select execution options

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
### Examples

```
python run.py -a OGD -t bc -d ./data/bc/
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
