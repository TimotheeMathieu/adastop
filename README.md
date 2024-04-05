# AdaStop

This package contains the AdaStop algorithm. AdaStop implements a *statistical test to adaptively choose the number of runs of stochastic algorithms* necessary to compare these algorithms and be able to rank them with a theoretically controlled family-wise error rate. One particular application for which AdaStop was created is to compare Reinforcement Learning algorithms. Please note, that what we call here *algorithm* is really *a certain implementation of an algorithm*.

The test proceeds in stages (or interims). First we collect $n$ performance measures for all $L$ algorithms computed on $n\times L$ different random seeds.
Then, Adastop examines these $n\times L$ numbers and decides that some of the algorithms are different, some of them are equal, and some of them needs more data to be distinguished. The process then repeats itself until a decision has been reached on all the algorithms.

The parameters of Adastop are described below, most important are $n$ the number of evaluations at each interim and $K$ the maximum number of interims.

# Installation

To install adastop, use pip:
```
pip install adastop
```

This will automatically install the command line interface as well as the python library.

WARNING: this Readme is for the dev version of adastop, to see the README associated to the released version, see https://pypi.org/project/adastop/

# Usage

There are two ways to use this package:

- Command line interface: AdaStop can be used as a command line interface that takes csv files as input. The cli interface can either be called interactively or the process can be automated using bash script.
- Python API: AdaStop is coded in python and can directly be imported as a module to be used in a python script.

Refer to [the documentation](https://timotheemathieu.github.io/adastop/) and in particular our [tutorial](https://timotheemathieu.github.io/adastop/tutorials.html) for detailed instructions on using `adastop`.


# Citation

AdaStop was originally developped for the article [AdaStop: adaptive statistical testing for sound comparisons of Deep RL agents](https://arxiv.org/abs/2306.10882) by Timothée Mathieu, Riccardo Della Vecchia, Alena Shilova, Matheus Medeiros Centa, Hector Kohler, Odalric-Ambrym Maillard, Philippe Preux.
