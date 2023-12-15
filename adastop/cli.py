import click
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
@click.option("--n-groups", default=5, show_default=True, help="Number of groups.")
@click.option("--n-permutations", default=10000, show_default=True, help="Number of random permutations.")
@click.option("--alpha", default=0.05, show_default=True, help="Type I error.")
@click.option("--beta", default=0.0, show_default=True, help="early accept parameter.")
@click.option("--seed", default=None, type=int, show_default=True, help="Random seed.")
@click.option("--compare-to-first", is_flag=True, show_default=True, default=False, help="Compare all algorithms to the first algorithm.")
@click.argument('input_file',required = True, type=str)
@click.pass_context
def compare(ctx, input_file, n_groups, n_permutations, alpha, beta, seed, compare_to_first):
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
        if len(Z[0]) > comparator.K * n_fits_per_group:
            raise ValueError('Error: you tried to use more group than what was initially declared, this is not allowed by the theory.')
        assert "continue" in list(comparator.decisions.values()), "Test finished at last iteration."

    else:
        comparator = MultipleAgentsComparator(n_fits_per_group, n_groups,
                                              n_permutations, comparisons,
                                              alpha, beta, seed)
        Z = [df[agent].values for agent in df.columns]

    data = {df.columns[i] : Z[i] for i in range(len(df.columns))}
    # recover also the data of agent that were decided.
    if comparator.agent_names is not None:
        for agent in comparator.agent_names:
            if agent not in df.columns:
                data[agent]=comparator.eval_values[agent]

    comparator.partial_compare(data, False)
    if not("continue" in list(comparator.decisions.values())):
        click.echo('')
        click.echo("Test is finished, decisions are")
        click.echo(comparator.get_results().to_markdown())
        
    else:
        still_here = []
        for c in comparator.comparisons:
            if comparator.decisions[str(c)] == "continue":
                still_here.append( comparator.agent_names[c[0]])
                still_here.append( comparator.agent_names[c[1]])
        still_here = np.unique(still_here)
        click.echo("Still undecided about "+" ".join(still_here))
    click.echo('') 
    
    with open(path_lf, 'wb') as fp:
        pickle.dump(comparator, fp)
        click.echo("Comparator Saved")

@adastop.command()
@click.argument('folder',required = True, type=str)
@click.pass_context
def reset(ctx, folder):
    """
    Reset the comparator to zero by removing the save file of the comparator situated in the folder 'folder'.
    """
    path_lf = Path(folder) / LITTER_FILE
    if os.path.isfile(path_lf):
        os.remove(path_lf)
        click.echo("Comparator file have been removed.")
    else:
        click.echo("no comparator file found.")



@adastop.command()
@click.argument('folder',required = True, type=str)
@click.pass_context
def plot(ctx, folder):
    """
    Plot results of the comparator situated in the folder 'folder'.
    """
    path_lf = Path(folder) / LITTER_FILE
    if os.path.isfile(path_lf):
        with open(path_lf, 'rb') as fp:
            comparator = pickle.load(fp)
        assert not("continue" in list(comparator.decisions.values())), "Testing process not finished yet, cannot plot yet."
    else:
        raise ValueError('Comparator save file not found.')
    
    comparator.plot_results()
    plt.ion()
    plt.show()
    
