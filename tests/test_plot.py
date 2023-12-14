import pytest
from adastop import MultipleAgentsComparator
import numpy as np

B = 500
alpha = 0.05
n_runs = 10
K = 5
n = 4

def test_plot():
    n_agents = 3
    comparator = MultipleAgentsComparator(n=n, K=K, B=B,  alpha=alpha, seed=42, beta = 0.01)
    evals = {}
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] ,np.random.normal(size=n)])
        else:
            evals = {"Agent "+str(k): np.random.normal(size=n) for k in range(n_agents)}
        comparator.partial_compare(evals)
    comparator.plot_results()
