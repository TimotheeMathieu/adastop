# AdaStop -- guix
Sequential testing for efficient and reliable comparison of stochastic algorithms.

This package contains the AdaStop algorithm. AdaStop implements a *statistical test to adaptively choose the number of runs of stochastic algorithms* necessary to compare these algorithms and be able to rank them with a theoretically controlled family-wise error rate. One particular application for which AdaStop was created is to compare Reinforcement Learning algorithms. Please note, that what we call here *algorithm* is really *a certain implementation of an algorithm*.

The test proceed in stages (or interims). First we collect $n$ performance measures for all $L$ algorithms computed on $n\times L$ different random seeds.
Then, Adastop examines these $n\times L$ numbers and decides that some of the algorithms are different, some of them are equal, and some of them needs more data to be distinguished. The process then reapeats until a decision have been reached on all the algorithms.

The parameters of Adastop are described below, most important are $n$ the number of evaluations at each interim and $K$ the maximum number of interims.

# Installation

we provide a guix channel that can be used with guix container to have a reproducible adastop execution. 
```
curl -C - https://raw.githubusercontent.com/TimotheeMathieu/adastop/guix/channels.scm > channels.scm
guix time-machine --channels=channels.scm -- shell -CN python-adastop -- adastop compare my_file.csv
```
Remark that with these command line, nothing is installed and the computation is done in a container. You can use `guix gc` to clean up the necessary packages downloaded by guix during the above command.
