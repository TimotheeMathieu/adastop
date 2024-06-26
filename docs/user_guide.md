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

## AdaStop 

The advised way to use adastop is via its command line interface. We explain here the usage of each of `adastop`'s commands.

### Invoking `adastop compare`

The command `adastop compare` does one interim of AdaStop algorithm, i.e. it analyses one batch of data and decide whether to continue (keep gathering further data) or whether the test is decided.

As input, `adastop compare` takes a csv file containing as many lines as there are scores (plus one line for headers) and as many columns as there are algorithms to compare (plus one for the run numbers). At the end of the process, you may have to generate up to `n-groups` csv files, typically 5 to 10 files. For example, to compare a file with 5 scores, while aiming for a maximum of 6 interims, with a family-wise error of $0.05$, use

```bash
adastop compare --size-group 5 --n-groups 6 --alpha 0.05 first_results.csv
```

This command will create a hidden file `.adastop_comparator.pkl` that contains the current state of the comparator. Remark: if you want to reset the comparator, you may call `adastop reset` with argument the folder containing the hidden file, typically you may want to do `adastop reset .`.

Then, once you did the comparison on the first file, you can use iteratively `adastop compare` to continue the comparison on further data. See the [tutorial](Tutorial) for an example of use.

### Invoking `adastop plot`

The command `adastop plot` generate a plot representing the results of the comparison of `adastop`. It can only be done once `adastop compare` has been executed until completion. Then, to plot the result of a comparison, use 

```bash
adastop plot DIR output_file.pdf 
```
where DIR is the directory in which the hidden `.adastop_comparator.pkl` file is located, i.e. it is the directory in which you did `adastop compare`, typically DIR will be the current directory `.`. `output_file.pdf` is the file in which to export the plot, the format of the file will be guessed from the suffix, `png`, `jpg` and `pdf` are accepted.

You can also specify the height and width of the output graph using the `width` and `height` parameters for the command.
