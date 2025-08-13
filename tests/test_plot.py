import pytest
from adastop import MultipleAgentsComparator
import numpy as np
import matplotlib.pyplot as plt
B = 500
alpha = 0.05
n_runs = 10
K = 5
n = 4
seed = 42

def test_plot():
    n_agents = 3
    comparator = MultipleAgentsComparator(n=n, K=K, B=B,  alpha=alpha, seed=42)
    evals = {}
    rng = np.random.RandomState(seed)
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] ,rng.normal(size=n)])
        else:
            evals = {"Agent "+str(k): rng.normal(size=n) for k in range(n_agents)}
        comparator.partial_compare(evals)
    comparator.plot_results()

    # plt.savefig('fig.pdf')
    fig, axes= plt.subplots(1,2)
    comparator.plot_results(axes=axes)

    
def test_plot_sota():
    n_agents = 3
    rng = np.random.RandomState(seed)

    comparisons = np.array([(0,i) for i in [1,2]])
    comparator = MultipleAgentsComparator(n=n, K=K, B=B,  alpha=alpha, 
                                          comparisons=comparisons, seed=42)
    evals = {}
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] ,rng.normal(size=n)])
        else:
            evals = {"Agent "+str(k): rng.normal(size=n) for k in range(n_agents)}
        comparator.partial_compare(evals)
    comparator.plot_results_sota()
    # plt.savefig('fig2.pdf')
    fig, axes= plt.subplots(1,2)
    comparator.plot_results_sota(axes=axes)

def test_plot_noteq():
    n_agents = 3
    comparator = MultipleAgentsComparator(n=10, K=K, B=B,  alpha=alpha, seed=42)
    rng = np.random.RandomState(seed)

    evals = {}
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] , k+ rng.normal(size=10)])
        else:
            evals = {"Agent "+str(k): rng.normal(size=10)+k for k in range(n_agents)}
        comparator.partial_compare(evals)
    # plt.savefig('fig2.pdf')
    fig, axes= plt.subplots(1,2)
    comparator.plot_results(axes=axes)

def test_plot_sota_noteq():
    n_agents = 3
    rng = np.random.RandomState(seed)

    comparisons = np.array([(0,i) for i in [1,2]])
    comparator = MultipleAgentsComparator(n=10, K=K, B=B,  alpha=alpha, 
                                          comparisons=comparisons, seed=42)
    evals = {}
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] , rng.normal(size=10)+k])
        else:
            evals = {"Agent "+str(k): rng.normal(size=10) for k in range(n_agents)}
        comparator.partial_compare(evals)
    comparator.plot_results_sota()
    # plt.savefig('fig2.pdf')
    fig, axes= plt.subplots(1,2)
    comparator.plot_results_sota(axes=axes)



def test_plot_noteq2():
    n_agents = 3
    comparator = MultipleAgentsComparator(n=10, K=K, B=B,  alpha=alpha, seed=42)
    rng = np.random.RandomState(seed)

    evals = {}
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] , np.abs(2*K-k)+ rng.normal(size=10)])
        else:
            evals = {"Agent "+str(k): rng.normal(size=10)+np.abs(2*K-k) for k in range(n_agents)}
        comparator.partial_compare(evals)
    # plt.savefig('fig2.pdf')
    fig, axes= plt.subplots(1,2)
    comparator.plot_results(axes=axes)

def test_plot_sota_noteq2():
    n_agents = 3
    comparisons = np.array([(0,i) for i in [1,2]])
    comparator = MultipleAgentsComparator(n=10, K=K, B=B,  alpha=alpha, 
                                          comparisons=comparisons, seed=42)
    rng = np.random.RandomState(seed)

    evals = {}
    while not comparator.is_finished:
        if len(evals) >0:
            for k in range(n_agents):
                evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] , rng.normal(size=10)+np.abs(2*K-k)])
        else:
            evals = {"Agent "+str(k): rng.normal(size=10)+np.abs(2*K-k) for k in range(n_agents)}
        comparator.partial_compare(evals)
    comparator.plot_results_sota()
    # plt.savefig('fig2.pdf')
    fig, axes= plt.subplots(1,2)
    comparator.plot_results_sota(axes=axes)

