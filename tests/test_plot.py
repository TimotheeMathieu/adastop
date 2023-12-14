import pytest
from adastop import MultipleAgentsComparator
import numpy as np
    
B = 5000
alpha = 0.05
n_runs = 10
K = 5
n = 4

def test_plot():
    idxs = []
    n_agents = 3
    for M in range(n_runs):
        comparator = MultipleAgentsComparator(n=n, K=K, B=B,  alpha=alpha, seed=M, beta = 0.01)
        evals = {}
        while not comparator.is_finished:
            if len(evals) >0:
                for k in range(n_agents):
                    evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] ,np.random.normal(size=n)])
            else:
                evals = {"Agent "+str(k): np.random.normal(size=n) for k in range(n_agents)}
            comparator.partial_compare(evals)
        idxs.append(not("equal" in comparator.decisions.values()))
    comparator.plot_results()
