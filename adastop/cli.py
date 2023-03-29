import click
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np

from .compare_agents import MultipleAgentsComparator

LITTER_FILE = ".adastop_comparator.pkl"

@click.group()
@click.pass_context
def adastop(ctx):
    """
    Program to perform adaptive stopping algorithm using csv file intput_file.

    Use adastop sub-command --help to have help for a specific sub-command
    """
    pass

@adastop.command()
@click.option("--n-groups", default=5, help="Number of groups.")
@click.option("--n-permutations", default=10000, help="Number of random permutations.")
@click.option("--alpha", default=0.1, help="Type I error.")
@click.option("--beta", default=0.01, help="early accept parameter.")
@click.option("--compare-to-first", is_flag=True, show_default=True, default=False, help="Compare all agents to the first agent.")
@click.argument('input_file',required = True, type=str)
@click.pass_context
def compare(ctx, input_file, n_groups, n_permutations, alpha, beta, compare_to_first):
    """
    Perform one step of adaptive stopping algorithm using csv file intput_file.
    At first call, the comparator will be initialized with the arguments passed and then it will be saved to a save file in `.adastop_comparator.pkl`.
    """
    path_lf = Path(input_file).parent.absolute() / LITTER_FILE
    df = pd.read_csv(input_file, index_col=0)
    n_fits_per_group = len(df) 
    n_agents = len(df.columns)
    if compare_to_first:
        comparisons = [(0,i) for i in range(1, n_agents)]
    else:
        comparisons = None

    # if this is not first group, load data for comparator.
    if os.path.isfile(path_lf):
        with open(path_lf, 'rb') as fp:
            comparator = pickle.load(fp)
        Z = [np.hstack([comparator.eval_values[agent], df[agent]]) for agent in df.columns]
        if len(Z[0])> n_groups * n_fits_per_group:
            raise ValueError('Error: you tried to use more group than what was initially declared, this is not allowed by the theory.')
        assert "continue" in list(comparator.decisions.values()), "Test finished at last iteration."

    else:
        comparator = MultipleAgentsComparator(n_fits_per_group, n_groups, n_permutations, comparisons, alpha, beta)
        Z = [df[agent].values for agent in df.columns]

    comparator.partial_compare({df.columns[i] : Z[i] for i in range(len(df.columns))}, True)
    if not("continue" in list(comparator.decisions.values())):
        click.echo('')
        click.echo("Test is finished, decisions are")
    click.echo('')
    for c in comparator.comparisons:
        click.echo("Decision between "+ comparator.agent_names[c[0]] +" and "+comparator.agent_names[c[1]]+ " is: " +comparator.decisions[str(c)])

    click.echo('')

    
    with open(path_lf, 'wb') as fp:
        pickle.dump(comparator, fp)
        click.echo("Comparator Saved")

@adastop.command()
@click.pass_context
def reset():
    """
    Reset the comparator to zero by removing the save file of the comparator.
    """
    path_lf = Path(input_file).parent.absolute() / LITTER_FILE
    os.remove(path_lf)


@adastop.command()
@click.pass_context
def plot():
    """
    Plot results of the comparator.
    """
    path_lf = Path(input_file).parent.absolute() / LITTER_FILE
    if os.path.isfile(path_lf):
        with open(path_lf, 'rb') as fp:
            comparator = pickle.load(fp)
        assert not("continue" in list(comparator.decisions.values())), "Testing process not finished yet, cannot plot yet."
    else:
        raise ValueError('Comparator save file not found.')
    
    comparator.plot_results()
    
if __name__ == '__main__':
    adastop()
