import numpy as np
from scipy.optimize import bisect, minimize_scalar
from copy import deepcopy

class Bounded_Confseq():
    """
    Compute a confidence sequence for bounded data using the hedged betting confidence sequence from 
    Section 4.4 of Estimating means of bounded random variables by betting. JOURNAL OF THE ROYAL STATISTICAL SOCIETY SERIES B
    by Waudby-Smith and Ramdas (2022)

    Parameters
    ----------

    delta: float, default=0.05
        Type I error. The resulting confidence interval will be of level 1-delta
    
    bounds: tuple of floats, default=(0,1)
        Bounds on the support of the distribution of values. 

    c: float, default=3/4
        Tuning parameter for the betting process, measure how conservative the confidence sequence is. 3/4 is a good default.


    Attributes
    ----------

    self.stats: dictionary with keys "est": the estimation, "lb": lower bound of confidence sequence, "ub": upper bound of the confidence sequence and "N": number of samples.

    Example
    -------

    To be used only on data that are bounded between 0 and 1:

    >>> CS = Bounded_Confseq()
    >>>
    >>> for x in observations:
    >>>     CS.observe(x)
    >>>     print(CS.stats)

    the resulting confidence sequence found in CS.stats["lb"] and CS.stats["ub"] contains, at any time, the mean of the distribution with probability larger than 1-delta.
    """
    def __init__(self, delta = 0.05, bounds = (0,1), c = 3/4):
        self.bounds = bounds
        self.c = c
        self.delta = delta
        self.mu = 1/2
        self.sigma2 = 1/4
        self.eff_ss = 1
        self.stats = {"est": np.array([]),
                     "lb" : [],
                      "ub": [],
                      "N": []}
        self.lambdas = np.array([])
        self.X = np.array([])

    
    def observe(self, x):
        x = (x - self.bounds[0])/(self.bounds[1]-self.bounds[0])
        if self.eff_ss >1:
            lamb = np.sqrt(2 * np.log(2/self.delta)/(self.sigma2 * (self.eff_ss-1) * np.log(self.eff_ss)))
        else:
            lamb = 0

        self.lambdas = np.append(self.lambdas, min(lamb, self.c))
        self.X = np.append(self.X, x)

        ret = minimize_scalar(lambda m : np.log(self.capital(m)) ,bounds= [0,1], method="bounded")
        
        if (np.log(self.capital(ret.x)) > np.log(1/self.delta)) or (np.log(self.capital(0)) < np.log(1/self.delta)) :
            lb = 0
        else:
            lb = bisect(lambda m : np.log(self.capital(m)) - np.log(1/self.delta), 0, ret.x)
            
        if (np.log(self.capital(1)) < np.log(1/self.delta)) or (np.log(self.capital(ret.x)) > np.log(1/self.delta)) :
            ub = 1
        else:
            ub = bisect(lambda m : np.log(self.capital(m)) - np.log(1/self.delta), ret.x, 1)

        lb = lb *(self.bounds[1]-self.bounds[0]) + self.bounds[0]
        ub = ub *(self.bounds[1]-self.bounds[0]) + self.bounds[0]

        self.stats["est"] = np.append(self.stats["est"], (lb+ub)/2)

        
        if self.eff_ss > 1:
            self.stats["lb"].append(max(lb, self.stats["lb"][-1]))
            self.stats["ub"].append(min(ub, self.stats["ub"][-1]))
        else:
            self.stats["lb"].append(lb)
            self.stats["ub"].append(ub)

        self.mu = (self.eff_ss * self.mu + x) / (self.eff_ss + 1)
        self.sigma2 = (self.eff_ss * self.sigma2 + (x-self.mu)**2) / (self.eff_ss + 1)
        self.eff_ss += 1
            

    def capital(self, m):
        if (m == 0) or (m == 1):
            return np.inf

        else:
            lambp = deepcopy(self.lambdas)
            lambp[lambp > self.c/m] = self.c/m

            lambm = deepcopy(self.lambdas)
            lambm[lambm > self.c/(1-m)] = self.c/(1-m)

            Kp = np.prod(1 + lambp * (self.X -m))
            Km = np.prod(1 - lambm * (self.X -m))
            if (Kp  + Km >0):
                return (Kp+ Km)/2
            else:
                return 1e-10
