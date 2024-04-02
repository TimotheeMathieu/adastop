# AdaStop

Sequential testing for efficient and reliable comparison of stochastic algorithms.

This package contains the AdaStop algorithm. AdaStop implements a sequential statistical test using group sequential permutation test and is especially adapted to multiple testing with very small sample size.

We use AdaStop in particular to *to adaptively choose the number of runs of fully specified algorithms with stochastic returns* and to get a *statistically significant decision* on a comparison of algorithms. The rationale is that when the returns of some experiment in computer science is stochastic, it becomes necessary to make the same experiment several time in order to have a viable comparison of the algorithms and be able to rank them with a theoretically controlled family-wise error rate.  Adastop allows us to choose the number of repetition adaptively to stop collecting data as soon as possible. One particular application for which AdaStop was created is to compare Reinforcement Learning algorithms. Please note, that what we call here *algorithm* is really *a certain implementation of an algorithm*.



## Tutorial

Basic usage of AdaStop through example.

```{toctree}
:maxdepth: 2

tutorials
```


## User Guide


Information on how to use AdaStop library.

```{toctree}
:maxdepth: 2

user_guide
```


## API

Information on python classes behind AdaStop.

```{toctree}
:maxdepth: 2

api
```
