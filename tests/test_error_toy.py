import pytest
from adastop import MultipleAgentsComparator
import numpy as np
    
B = 5000
alpha = 0.05
n_runs = 10

def test_runtime():
    idxs = []
    comparator = MultipleAgentsComparator(n=3, K=3, B=B,  alpha=alpha, seed=42, beta = 0.01, joblib_backend = "multiprocessing")
    evals = {"Agent "+str(k):np.random.normal(size=3) for k in range(3)}
    comparator.partial_compare(evals)
    

@pytest.mark.parametrize("K,n", [(5,3), (3, 5), (1, 15)])
def test_type1(K,n):
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
        print(comparator.get_results())
    assert np.mean(idxs) < 2*alpha + 1/4/(np.sqrt(n_runs)), "type 1 error seems to be too large."
        
@pytest.mark.parametrize("K,n", [(3, 5), (1, 15)])
def test_type2(K,n):
    idxs = []
    n_agents = 2
    for M in range(n_runs):
        comparator = MultipleAgentsComparator(n=n, K=K, B=B,  alpha=alpha, seed=M, beta = 0.01)
        evals = {}
        while not comparator.is_finished:
            if len(evals) >0:
                for k in range(n_agents):
                    evals["Agent "+str(k)] = np.hstack([evals["Agent "+str(k)] ,np.random.normal(size=n)+2*k])
            else:
                evals = {"Agent "+str(k): np.random.normal(size=n)+2*k for k in range(n_agents)}
            comparator.partial_compare(evals)
        idxs.append(not("equal" in comparator.decisions.values()))
    assert np.mean(idxs) > 0.3, "type 2 error seems to be too large."
        
