import logging
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import math
import pandas as pd
from joblib import Parallel, delayed
import itertools
from .plotting import plot_results, plot_results_sota

logger = logging.getLogger()


# TODO:
# - make a get_result function here that also print a textual summary.

class MultipleAgentsComparator:
    """
    Compare sequentially agents, with possible early stopping.
    At maximum, there can be n times K fits done.

    For now, implement only a two-sided test.

    Parameters
    ----------

    n: int, or array of ints of size self.n_agents, default=5
        If int, number of fits before each early stopping check. If array of int, a
        different number of fits is used for each agent.

    K: int, default=5
        number of check.
    
    B: int, default=None
        Number of random permutations used to approximate permutation distribution.
    
    comparisons: list of tuple of indices or None
        if None, all the pairwise comparison are done.
        If = [(0,1), (0,2)] for instance, the compare only 0 vs 1  and 0 vs 2
    
    alpha: float, default=0.01
        level of the test

    beta: float, default=0
        power spent in early accept.

    e_values: boolean, default=False
        Whether to use e-values of p-values.
    
    seed: int or None, default = None

    joblib_backend: str, default = "threading"
        backend to use to parallelize on multi-agents. Use "multiprocessing" or "loky" for a true parallelization.

    Attributes
    ----------
    agent_names: list of str
        list of the agents' names.
    decision: dict
        decision of the tests for each comparison, keys are the comparisons and values are in {"equal", "larger", "smaller"}.
    n_iters: dict
        number of iterations (i.e. number of fits) used for each agent. Keys are the agents' names and values are ints.
    
    Examples
    --------
    One can either use rlberry with self.compare, pre-computed scalars with self.compare_scalar or one can use
    the following code compatible with basically anything:

    >>> comparator = MultipleAgentsComparator(n=6, K=6, B=10000, alpha=0.05)
    >>>
    >>> eval_values = {agent.name: [] for agent in agents}
    >>>
    >>> for k in range(comparator.K):
    >>>    for  agent in enumerate(agents):
    >>>        # If the agent is still in one of the comparison considered, then generate new evaluations.
    >>>        if agent in comparator.current_comparisons.ravel():
    >>>            eval_values[agent.name].append(train_evaluate(agent, n))
    >>>    comparator.partial_compare(eval_values, verbose)
    >>>    decisions = comparator.decisions # results of the decisions for step k
    >>>    if comparator.is_finished:
    >>>        break

    Where train_evaluate(agent, n) is a function that trains n copies of agent and returns n evaluation values.
    """

    def __init__(
        self,
        n=5,
        K=5,
        B=10000,
        comparisons = None,
        alpha=0.01,
        beta=0,
        e_values=False,
        seed=None,
        joblib_backend="threading",
    ):
        self.n = n
        self.K = K
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.e_values = e_values
        self.comparisons = comparisons
        self.boundary = []
        self.k = 0
        self.level_spent = 0
        self.power_spent = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.rejected_decision = []
        self.joblib_backend = joblib_backend
        self.agent_names = None
        self.current_comparisons = copy(comparisons)
        self.n_iters = None
        self.mean_diffs = None
        self.id_tracked = None
        self.is_finished = False

    def compute_mean_diffs(self, k, Z):
        """
        Compute the absolute value of the sum differences.
        """
        comparisons = self.current_comparisons
        boundary = self.boundary
        if k == 0:
            for i, comp in enumerate(comparisons):
                # Define set of permutations. Either all permutations or a random sample if all permutations
                # is too many permutations
                n1 = self.n[comp[0]]
                n2 = self.n[comp[1]]
                n_permutations = math.comb(n1+n2, n1)

                if self.B > n_permutations:
                    permutations = itertools.combinations(np.arange(n1+n2), n1)
                    self.normalization = n_permutations
                else:
                    permutations = [np.arange(n1)]+[ self.rng.permutation(n1+n2)[:n1] for _ in range(self.B-1)]
                    self.normalization = self.B

                # Compute the meajn differences on the evaluations for each comparisions and for each permutation
                mean_diff = []
                for perm in permutations:
                    Zi = np.hstack([Z[comp[0]][: n1], Z[comp[1]][: n2]])
                    mask = np.zeros(n1+n2)
                    mask[list(perm)] = 1
                    mask = mask == 1
                    mean_diff.append(np.mean(Zi[mask]) - np.mean(Zi[~mask]))
                self.mean_diffs[str(comp)] = mean_diff
        else:
            # Eliminate for conditional
            mean_diffs = {str(comp):[] for comp in self.mean_diffs}
            to_remove = []
            for i in range(len(self.mean_diffs[str(comparisons[0])])):
                zval = []
                for comp in self.mean_diffs:
                    zval.append(self.mean_diffs[str(comp)][i])
                if np.max(np.abs(zval)) <= boundary[-1][1]:
                    for comp in self.mean_diffs:
                        mean_diffs[str(comp)].append(self.mean_diffs[str(comp)][i])
                    
            for i, comp in enumerate(comparisons):

                n1 = self.n[comp[0]]
                n2 = self.n[comp[1]]
                
                # add new permutations. Can be either all the permutations of block k, or using random permutations if this is too many.
                n_permutations = math.comb(n1+n2, n1)
                if self.B > n_permutations ** (k+1):
                    permutations_k = itertools.combinations(np.arange(n1+n2), n1)
                    self.normalization = n_permutations ** (k+1) # number of permutations
                else :
                    n_perm_to_add = len(mean_diffs[str(comparisons[0])])
                    permutations_k = [np.arange(n1)]+[self.rng.permutation(n1+n2)[:n1] for _ in range(n_perm_to_add-1)]
                    began_random_at = np.floor(np.log(self.B)/np.log(n_permutations))
                    self.normalization = n_permutations ** began_random_at # number of permutations

                Zk = np.zeros(n1+n2)
                new_mean_diffs=[]
                for id_p, perm_k in enumerate(permutations_k):
                    if self.B > n_permutations ** (k+1):
                        perms_before_k = np.arange(len(mean_diffs[str(comparisons[0])])).astype(int)
                    else:
                        perms_before_k = [id_p]
                        
                    for perm_before_k in perms_before_k:
                        # Compute the mean diffs for given permutation and comparison i

                        Zk[:n1]=Z[comp[0]][(k * n1) : ((k + 1) * n1)]
                        Zk[n1:(n1+n2)] = Z[comp[1]][(k * n2) : ((k + 1) * n2)]
                        mask = np.zeros(n1+n2)
                        mask[list(perm_k)] = 1
                        mask = mask == 1

                        new_mean_diffs.append(mean_diffs[str(comp)][perm_before_k] + np.mean(Zk[mask]) - np.mean(Zk[~mask]))

                self.mean_diffs[str(comp)] = new_mean_diffs
        return self.mean_diffs

    def partial_compare(self, eval_values, verbose=True):
        """
        Do the test of the k^th interim.

        Parameters
        ----------
        eval_values: dict of agents and evaluations
            keys are agent names and values are concatenation of evaluations till interim k,
            e.g. {"PP0": [1,1,1,1,1], "SAC": [42,42,42,42,42]}
        verbose: bool
            print Steps
        Returns
        -------
        decisions: dictionary with comparisons as index and with values str in {"equal", "larger", "smaller", "continue"}
           Decision of the test at this step.
        id_finished: bool
           Whether the test is finished or not.
        T: float
           Test statistic.
        bk: float
           Thresholds of the tests.
        """
        if self.agent_names is None:
            self.agent_names = list(eval_values.keys())

        Z = [eval_values[agent] for agent in self.agent_names]
        n_managers = len(Z)
        if isinstance(self.n,int):
            self.n = np.array([self.n]*n_managers)
        
        if self.k == 0:
            # initialization
            if self.comparisons is None:
                self.comparisons = np.array(
                    [(i, j) for i in range(n_managers) for j in range(n_managers) if i < j]
                )
            self.current_comparisons = copy(self.comparisons)
            self.mean_diffs = {str(comp):[] for comp in self.comparisons}
            
            self.n_iters = {self.agent_names[i] : 0 for i in range(n_managers)}
                
            self.decisions = {str(c):"continue" for c in self.comparisons}
            self.id_tracked = np.arange(len(self.decisions)) # ids of comparisons effectively tracked

        
        k = self.k

        clevel = self.alpha*(k + 1) / self.K
        dlevel = self.beta*(k + 1) / self.K

        mean_diffs = self.compute_mean_diffs(k, Z)
        
        
        if verbose:
            print("Step {}".format(k))

        current_decisions = np.array(["continue"]*len(self.id_tracked))
        current_sign = np.zeros(len(current_decisions))
        
        for j in range(len(current_decisions)):
            current_comparisons = self.current_comparisons[current_decisions=="continue"]
            mean_diffs_now = { str(comp):mean_diffs[str(comp)] for comp in current_comparisons}
            max_mean_diffs = [ np.max([mean_diffs_now[str(comp)][i] for comp in current_comparisons]) for i in range(len(mean_diffs_now[str(current_comparisons[0])]))]
            values = np.sort(max_mean_diffs)  

            icumulative_probas = np.arange(len(values))[::-1] / self.normalization  # This corresponds to 1 - F(t) = P(T > t)

            # Compute admissible values, i.e. values that would not be rejected nor accepted.
            admissible_values_sup = values[
                self.level_spent + icumulative_probas <= clevel
            ]

            if len(admissible_values_sup) > 0:
                bk_sup = admissible_values_sup[0]  # the minimum admissible value
                level_to_add = icumulative_probas[
                    self.level_spent + icumulative_probas <= clevel
                ][0]
            else:
                # This case is possible if clevel-self.level_spent <= 1/ self.normalization (smallest proba possible),
                # in which case there are not enough points and we don't take any decision for now. Happens in particular if B is None.
                bk_sup = np.inf
                level_to_add = 0

            cumulative_probas = np.arange(len(values)) / self.normalization  # corresponds to P(T < t)
            admissible_values_inf = values[
                self.power_spent + cumulative_probas < dlevel
            ]

            if len(admissible_values_inf) > 0:
                bk_inf = admissible_values_inf[-1]  # the maximum admissible value
                power_to_add = cumulative_probas[
                    self.power_spent + cumulative_probas <= dlevel
                ][-1]
            else:
                bk_inf = -np.inf
                power_to_add = 0

            # Test statistic, step-down
            Tmax = 0
            Tmin = np.inf
            Tmaxsigned = 0
            Tminsigned = 0
            for i, comp in enumerate(self.current_comparisons[current_decisions == "continue"]):
                Ti = (k+1)*np.abs(
                    np.mean(
                        Z[comp[0]][: ((k + 1) * self.n[comp[0]])])
                        - np.mean(Z[comp[1]][: ((k + 1) * self.n[comp[1]])]
                    )
                )
                if Ti > Tmax:
                    Tmax = Ti
                    imax = i
                    Tmaxsigned = (k+1)*(np.mean(
                        Z[comp[0]][: ((k + 1) * self.n[comp[0]])])
                        - np.mean(Z[comp[1]][: ((k + 1) * self.n[comp[1]])]
                    )
                )

                if Ti < Tmin:
                    Tmin = Ti
                    imin = i
                    Tminsigned = (k+1)*(np.mean(Z[comp[0]][: ((k + 1) * self.n[comp[0]])])
                                        - np.mean(Z[comp[1]][: ((k + 1) * self.n[comp[1]])]
                                  )
                                    )

            if Tmax > bk_sup:
                id_reject = np.arange(len(current_decisions))[current_decisions== "continue"][imax]
                current_decisions[id_reject] = "reject"

                if Tmaxsigned >0:
                    self.decisions[str(self.current_comparisons[id_reject])] = "larger"
                else:
                    self.decisions[str(self.current_comparisons[id_reject])] = "smaller"
                if verbose:
                    print("reject")
            elif Tmin < bk_inf:
                id_accept = np.arange(len(current_decisions))[current_decisions == "continue"][imin]
                current_decisions[id_accept] = "accept"
                self.decisions[str(self.current_comparisons[id_accept])] = "equal"
            else:
                break

            
        
        self.boundary.append((bk_inf, bk_sup))

        self.level_spent += level_to_add  # level effectively used at this point
        self.power_spent += power_to_add 
        
        if k == self.K - 1:
            for c in self.comparisons:
                if self.decisions[str(c)]=="continue":
                    self.decisions[str(c)] = "equal"
        
        self.k = self.k + 1
        self.eval_values = {self.agent_names[i]: Z[i] for i in range(n_managers)}
        self.mean_eval_values = [np.mean(z) for z in Z]
        for i in range(n_managers):
            self.n_iters[self.agent_names[i]] = len(Z[i].ravel())

        id_decided = np.array(current_decisions) != "continue"
        
        if  not("continue" in self.decisions.values()):
            self.is_finished = True
            
        self.id_tracked = self.id_tracked[~id_decided]
        self.current_comparisons = self.current_comparisons[~id_decided]
        self.mean_diffs = {str(comp): self.mean_diffs[str(comp)] for comp in current_comparisons}

    def get_results(self):
        """
        Returns a dataframe with the results of the tests.
        """
        results = pd.DataFrame()
        for c in self.comparisons:
            results = pd.concat([results, pd.DataFrame(
                {
                    "Agent1 vs Agent2": 
                        ["{0} vs {1}".format(self.agent_names[c[0]], self.agent_names[c[1]])],
                    "mean Agent1": self.mean_eval_values[c[0]],
                    "mean Agent2": self.mean_eval_values[c[1]],
                    "mean diff": self.mean_eval_values[c[0]]-self.mean_eval_values[c[1]],
                    "decisions": self.decisions[str(c)],
                }
            )])
        return results

    def plot_results(self, agent_names=None, axes=None):
        """
        visual representation of results.

        Parameters
        ----------
        agent_names : list of str or None
        axes : tuple of two matplotlib axes of None
             if None, use the following:
             `fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(6,5))`
        """
        plot_results(self, agent_names, axes)

    def plot_results_sota(self, agent_names=None, axes=None):
        """
        visual representation of results when the first agent is compared to all the others.

        Parameters
        ----------
        agent_names : list of str or None
        axes : tuple of two matplotlib axes of None
             if None, use the following:
             `fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(6,5))`
        """
        plot_results_sota(self, agent_names, axes)
