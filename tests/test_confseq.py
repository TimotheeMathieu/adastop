import pytest
from adastop import Bounded_Confseq
import numpy as np
    
B = 5000
alpha = 0.05
n_data = 100
n_runs = 10
seed = 42

def test_type1():
    rng = np.random.RandomState(seed)
    res = []
    for M in range(n_runs):
        cs = Bounded_Confseq()
        data = rng.uniform(size=n_data)
        is_in = True
        for x in data:
            cs.observe(x)
            if (1/2 >= cs.stats["ub"][-1]) or (1/2 <= cs.stats["lb"][-1]):
                is_in = False
                break
        res.append(~is_in)
    assert np.mean(res) < alpha , "type 1 error seems to be too large."
        
