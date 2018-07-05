# LIBOL Python

Library for Online Learning algorithms in Python 3
This project is based on LIBOL https://github.com/LIBOL/LIBOL by stevenhoi,
and contains some First Order and Second Order algorithms, as well as options for 
customizing these algorithms as needed (regularization, kernel trick, hyperparameters, ...)

## Getting Started

Clone the repository and install the dependencies listed in the file requirements.txt, these dependencies are:

numpy
oct2py
nose2

### Run Test

To verify the correct installation of the Library execute the following command:
```
python run.py
```
This command runs the Perceptron algorithm for the breast_cancer_scale dataset from LIBSMV datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
The results of this execution should be somethin like this:

-------------------------------------------------------------------------------
Dataset name:  ./data/test/bc/breast_cancer_scale.txt ( n= 683  d= 10  m= 2 )    nb of runs (permutations):  20
-------------------------------------------------------------------------------
Algorithm:  Perceptron
mistake rate:  0.0582 +/- 0.004
nb of updates:  39.75 +/- 2.7363
cpu time (seconds):  0.0056 +/- 0.0001
-------------------------------------------------------------------------------


### Select execution options

The main execution of anonline learning algorithm is made from the run.py file,
the execution of this file requires the following console parameters

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
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

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
