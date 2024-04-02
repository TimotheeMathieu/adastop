# User Guide
The test proceed in stages (or interims). First we collect $n$ performance measures for all $L$ algorithms computed on $n\times L$ different random seeds.
Then, Adastop examines these $n\times L$ numbers and decides that some of the algorithms are different, some of them are equal, and some of them needs more data to be distinguished. The process then reapeats until a decision have been reached on all the algorithms.

The parameters of Adastop are described below, most important are $n$ the number of evaluations at each interim and $K$ the maximum number of interims.


## Installation

To install adastop, use pip:
```bash
pip install adastop
```

This will automatically install the command line interface as well as the python library.

WARNING: this Readme is for the dev version of adastop, to see the README associated to the released version, see https://pypi.org/project/adastop/



## Usage

See [the tutorial](tutorials).
