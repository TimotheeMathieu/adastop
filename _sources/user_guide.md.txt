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

The advised way to use adastop is via its command line interface. We explain here the usage of each of `adastop`'s commands. A documentation is available via the `--help` command for adastop but also for each of its sub-command.

```bash
$ adastop --help
adastop --help        
Usage: adastop [OPTIONS] COMMAND [ARGS]...

  Program to perform adaptive stopping algorithm using csv file intput_file.

  Use adastop sub-command --help to have help for a specific sub-command

Options:
  --help  Show this message and exit.

Commands:
  compare  Perform one step of adaptive stopping algorithm using csv file...
  plot     Plot results of the comparator situated in the folder 'folder'.
  reset    Reset the comparator to zero by removing the save file of the...
```

### Invoking `adastop compare`

The command `adastop compare` does one interim of AdaStop algorithm, i.e. it analyses one batch of data and decide whether to continue (keep gathering further data) or whether the test is decided.

As input, `adastop compare` takes a csv file containing as many lines as there are scores (plus one line for headers) and as many columns as there are algorithms to compare (plus one for the run numbers). At the end of the process, you may have to generate up to `n-groups` csv files, typically 5 to 10 files. For example, to compare a file with 5 scores, while aiming for a maximum of 6 interims, with a family-wise error of $0.05$, use

```bash
adastop compare --size-group 5 --n-groups 6 --alpha 0.05 first_results.csv
```

This command will create a hidden file `.adastop_comparator.pkl` that contains the current state of the comparator. Remark: if you want to reset the comparator, you may call `adastop reset` with argument the folder containing the hidden file, typically you may want to do `adastop reset .`.

Then, once you did the comparison on the first file, you can use iteratively `adastop compare` to continue the comparison on further data. See the [tutorial](Tutorial) for an example of use.

#### Choice of comparisons

In adastop, one can choose which comparisons are done. The default is to do all the pairwise comparisons between two algorithms. In practice, it is sometimes sufficient to compare to only one of them, a benchmark, for this the `--compare-to-first` argument can be used. For a more fine-grained control on which comparison to do, the python API can take the comparisons as input.

**Remark**: it is not statistically ok to execute adastop several times and interpret the result as though it was only one test, if adastop is run several times this is multiple testing and some calibration has to be done. Instead, it is better to do all the comparisons at the same time, running the adastop algorithm only once, and adastop will handle the multiplicity of hypotheses by itself.

#### adastop compare help message

```bash
$ adastop compare --help
Usage: adastop compare [OPTIONS] INPUT_FILE

  Perform one step of adaptive stopping algorithm using csv file intput_file.
  At first call, the comparator will be initialized with the arguments passed
  and then it will be saved to a save file in `.adastop_comparator.pkl`.

Options:
  --n-groups INTEGER        Number of groups.  [default: 5]
  --n-permutations INTEGER  Number of random permutations.  [default: 10000]
  --alpha FLOAT             Type I error.  [default: 0.05]
  --seed INTEGER            Random seed.
  --compare-to-first        Compare all algorithms to the first algorithm.
  --help                    Show this message and exit.
```


### Invoking `adastop plot`

The command `adastop plot` generate a plot representing the results of the comparison of `adastop`. It can only be done once `adastop compare` has been executed until completion. Then, to plot the result of a comparison, use 

```bash
adastop plot DIR output_file.pdf 
```
where DIR is the directory in which the hidden `.adastop_comparator.pkl` file is located, i.e. it is the directory in which you did `adastop compare`, typically DIR will be the current directory `.`. `output_file.pdf` is the file in which to export the plot, the format of the file will be guessed from the suffix, `png`, `jpg` and `pdf` are accepted.

You can also specify the height and width of the output graph using the `width` and `height` parameters for the command.

#### adastop plot help message

```bash
$ adastop plot --help
Usage: adastop plot [OPTIONS] FOLDER TARGET_FILE

  Plot results of the comparator situated in the folder 'folder'.

Options:
  --width INTEGER
  --height INTEGER
  --help            Show this message and exit.

```

